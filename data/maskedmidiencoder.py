import numpy as np

from data.midiencoder import MidiEncoder
from data.multitokencoder import MultiTokEncoder


class MaskedMidiEncoder:
    def __init__(self, base_encoder: MultiTokEncoder | MidiEncoder, masking_probability: float = 0.15):
        self.base_encoder = base_encoder
        self.masking_probability = masking_probability

        self.vocab = [f"<SENTINEL_{idx}>" for idx in range(100)] + ["<MASK>"] + base_encoder.vocab

        # add midi tokens to vocab
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "MaskedMidiEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def mask_record(self, record: dict) -> tuple[list[str], list[str]]:
        """
        Mask record and return tuple of src and tgt tokens with masks.
        """
        src_tokens = self.base_encoder.tokenize_src(record)
        tgt_tokens = src_tokens.copy()
        num_masks = self.masking_probability * len(src_tokens)

        ids_to_mask = np.random.randint(len(src_tokens), size=int(num_masks))

        for idx in range(len(src_tokens)):
            if idx in ids_to_mask:
                src_tokens[idx] = "<MASK>"
            else:
                tgt_tokens[idx] = "<MASK>"

        return src_tokens, tgt_tokens

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
