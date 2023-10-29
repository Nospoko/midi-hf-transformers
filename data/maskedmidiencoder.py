import torch
import numpy as np

from data.midiencoder import MidiEncoder
from data.multitokencoder import MultiTokEncoder


class MaskedMidiEncoder:
    def __init__(self, base_encoder: MultiTokEncoder | MidiEncoder, masking_probability: float = 0.15):
        self.base_encoder = base_encoder
        self.masking_probability = masking_probability
        self.num_sentinel = 100

        self.vocab = [f"<SENTINEL_{idx}>" for idx in range(self.num_sentinel)] + ["<MASK>"] + base_encoder.vocab

        # add midi tokens to vocab
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "MaskedMidiEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def mask_record(self, record: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Mask record and return tuple of src and tgt tokens with masks.
        """
        src_tokens = self.base_encoder.tokenize_src(record)
        tgt_tokens = src_tokens.copy()
        num_masks = self.masking_probability * len(src_tokens)

        ids_to_mask = np.random.randint(len(src_tokens), size=int(num_masks))
        np_src = np.array(src_tokens)
        np_tgt = np.array(tgt_tokens)

        # create a tgt mask which will be the opposite of src masking
        tgt_mask = np.ones_like(np_tgt, dtype=bool)
        tgt_mask[ids_to_mask] = 0

        np_src[ids_to_mask] = "<MASK>"
        np_tgt[tgt_mask] = "<MASK>"

        return np_src, np_tgt

    def encode_record(self, record: dict) -> tuple[list[int], list[int]]:
        """
        Encode record into src and tgt for unsupervised T5 learning.
        """
        src_tokens, tgt_tokens = self.mask_record(record)
        mask_token_id = self.token_to_id["<MASK>"]

        src = [self.token_to_id[token] for token in src_tokens]
        tgt = [self.token_to_id[token] for token in tgt_tokens]

        src = self.replace_masks(token_ids=src, mask_id=mask_token_id)
        tgt = self.replace_masks(token_ids=tgt, mask_id=mask_token_id)

        return src, tgt

    @staticmethod
    def replace_masks(token_ids: list[int], mask_id):
        """
        Replace every sequence of <MASK> token with one of <SENTINEL_[idx]>.
        Sentinel tokens do not repeat inside one sequence.
        """
        new_list = []
        current_sequence = []  # Store the current sequence of masks
        sentinel_token = 0
        for idx in token_ids:
            if idx == mask_id:
                current_sequence.append(idx)
            else:
                if current_sequence:
                    new_list.append(sentinel_token)
                    sentinel_token += 1
                    current_sequence = []
                new_list.append(idx)

        # Check if the last sequence was <MASK>s
        if current_sequence:
            new_list.append(sentinel_token)
        return new_list

    def decode(self, src_token_ids: torch.Tensor, tgt_token_ids: torch.Tensor):
        src_it = 0
        tgt_it = 0
        token_ids = []
        while src_it < len(src_token_ids) and tgt_it < len(tgt_token_ids):
            if src_token_ids[src_it] < self.num_sentinel:
                while tgt_it < len(tgt_token_ids) and tgt_token_ids[tgt_it] >= self.num_sentinel:
                    token_ids.append(tgt_token_ids[tgt_it])
                    tgt_it += 1
                src_it += 1
            elif tgt_token_ids[tgt_it] < self.num_sentinel:
                while src_it < len(src_token_ids) and src_token_ids[src_it] >= self.num_sentinel:
                    token_ids.append(src_token_ids[src_it])
                    src_it += 1
                tgt_it += 1
            else:
                src_it += 1
                tgt_it += 1

        tokens = [self.vocab[token_id] for token_id in token_ids]
        return self.base_encoder.untokenize_src(tokens)
