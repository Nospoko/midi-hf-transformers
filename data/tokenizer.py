import itertools

import numpy as np
import pandas as pd
from omegaconf import DictConfig


# new name to keep classnames from previous repo unambiguous
class MultiTokEncoder:
    def __init__(self):
        self.token_to_id = None
        self.vocab = None

    def tokenize_src(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *tokenize_src* implementation")

    def untokenize_src(self, tokens: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Your encoder needs *untokenize_src* implementation")

    def tokenize_tgt(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *tokenize_tgt* implementation")

    def untokenize_tgt(self, tokens: list[str]) -> list[int]:
        raise NotImplementedError("Your encoder needs *untokenize_tgt* implementation")

    def decode_src(self, token_ids: list[int]) -> pd.DataFrame:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        df = self.untokenize_src(tokens)
        return df

    def decode_tgt(self, token_ids: list[int]) -> list[int]:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        values = self.untokenize_tgt(tokens)
        return values

    def encode_src(self, record: dict) -> list[int]:
        tokens = self.tokenize_src(record)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids

    def encode_tgt(self, record: dict) -> list[int]:
        tokens = self.tokenize_tgt(record)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids


class MultiVelocityEncoder(MultiTokEncoder):
    def __init__(self, quantization_cfg: DictConfig, keys: list[str] = None):
        super().__init__()
        self.quantization_cfg = quantization_cfg
        if keys is None:
            self.keys = ["pitch", "dstart_bin", "duration_bin", "velocity_bin"]
        else:
            self.keys = keys

        self.specials = ["<CLS>", "<PAD>"]

        self.vocab = list(self.specials)

        # add midi tokens to vocab
        self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "MultiVelocityEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self):
        time_tokens_product = itertools.product(
            # weird because we want to use dstart if there is a dstart_bin key, and start if there is a start_bin
            range(self.quantization_cfg[self.keys[1][:-4]]),
            range(self.quantization_cfg.duration),
        )

        for pitch in range(21, 109):
            self.vocab.append(f"PITCH-{pitch}")
        for start, duration in time_tokens_product:
            self.vocab.append(f"TIME-{start:0.0f}-{duration:0.0f}")
        # in hf T5 and BART we will use common vocabulary for tgt and src tokens ¯\_(ツ)_/¯
        # luckily, MultiTok is ideal for this
        # (UNLESS we will manually swap a final layer (somehow))
        # TODO: find out how sharing tokens effects model performance.
        #   (VEL-1, VEL-2, VEL-3 for 3 bins and also for 128 bins)
        for velocity in range(128):
            self.vocab.append(f"VEL-{velocity:0.0f}")

    def tokenize_src(self, record: dict) -> list[str]:
        tokens = []
        n_samples = len(record[self.keys[0]])
        for idx in range(n_samples):
            pitch_token = f"PITCH-{record[self.keys[0]][idx]:0.0f}"
            time_token = f"TIME-{record[self.keys[1]][idx]}-{record[self.keys[2]][idx]:0.0f}"
            velocity_token = f"VEL-{record[self.keys[3]][idx]:0.0f}"
            tokens += [pitch_token, time_token, velocity_token]

        return tokens

    def tokenize_tgt(self, record: dict) -> list[str]:
        tokens = [f"VEL-{velocity:0.0f}" for velocity in record["velocity"]]
        return tokens

    def untokenize_src(self, tokens: list[str]) -> pd.DataFrame:
        samples = []
        buff = []
        if tokens[0] == "<CLS>":
            # get rid of cls token
            tokens = tokens[1:]

        for idx, token in enumerate(tokens):
            if token == "<PAD>":
                break
            if idx % 3 == 0:
                # pitch token
                buff.clear()
                # [6:] to clear "PITCH-" prefix from a token
                buff.append(int(token[6:]))
            elif idx % 3 == 1:
                # time token
                # [5:] to clear "TIME-" prefix from the token
                buff += [int(txt) for txt in token[5:].split("-")]
            else:
                # velocity token
                # [4:] to clear "VEL-" prefix from the token
                buff.append(int(token[4:]))
                samples.append(buff)

        df = pd.DataFrame(samples, columns=self.keys)

        return df

    def untokenize_tgt(self, tokens: list[str]) -> list[int]:
        # [4:] to clear "VEL-" prefix from the end of a token
        velocities = [int(token[4:]) for token in tokens if token not in self.specials]

        return velocities


class MultiStartEncoder(MultiTokEncoder):
    def __init__(self, quantization_cfg: DictConfig, keys: list[str] = None, tgt_bins: int = 650):
        super().__init__()
        self.quantization_cfg = quantization_cfg
        self.tgt_bins = tgt_bins
        if keys is None:
            self.keys = ["pitch", "dstart_bin", "duration_bin", "velocity_bin"]
        else:
            self.keys = keys

        self.specials = ["<CLS>", "<PAD>"]

        self.vocab = list(self.specials)

        # add midi tokens to vocab
        self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "MultiStartEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self):
        time_tokens_product = itertools.product(
            # weird because we want to use dstart if there is a dstart_bin key, and start if there is a start_bin key
            range(self.quantization_cfg[self.keys[1][:-4]]),
            range(self.quantization_cfg.duration),
        )

        for pitch in range(21, 109):
            self.vocab.append(f"PITCH-{pitch}")
        for start, duration in time_tokens_product:
            self.vocab.append(f"TIME-{start:0.0f}-{duration:0.0f}")
        for velocity in range(self.quantization_cfg.velocity):
            self.vocab.append(f"VEL-{velocity:0.0f}")
        for start in range(self.tgt_bins):
            self.vocab.append(f"START-{start}")

    def tokenize_src(self, record: dict) -> list[str]:
        tokens = []
        n_samples = len(record[self.keys[0]])
        for idx in range(n_samples):
            pitch_token = f"PITCH-{record[self.keys[0]][idx]:0.0f}"
            time_token = f"TIME-{record[self.keys[1]][idx]}-{record[self.keys[2]][idx]:0.0f}"
            velocity_token = f"VEL-{record[self.keys[3]][idx]:0.0f}"
            tokens += [pitch_token, time_token, velocity_token]

        return tokens

    def quantize_start(self, starts: pd.Series):
        bin_edges = np.linspace(start=0, stop=len(starts), num=self.tgt_bins)

        quantized_starts = np.digitize(starts, bin_edges) - 1
        return quantized_starts

    def tokenize_tgt(self, record: dict) -> list[str]:
        quantized_starts = self.quantize_start(record["start"])
        tokens = [f"START-{start_bin:0.0f}" for start_bin in quantized_starts]
        return tokens

    def untokenize_src(self, tokens: list[str]) -> pd.DataFrame:
        samples = []
        buff = []
        if tokens[0] == "<CLS>":
            # get rid of cls token
            tokens = tokens[1:]

        for idx, token in enumerate(tokens):
            if token == "<PAD>":
                break
            if idx % 3 == 0:
                # pitch token
                buff.clear()
                # [6:] to clear "PITCH-" prefix from a token
                buff.append(int(token[6:]))
            elif idx % 3 == 1:
                # time token
                # [5:] to clear "TIME-" prefix from the token
                buff += [int(txt) for txt in token[5:].split("-")]
            else:
                # velocity token
                # [4:] to clear "VEL-" prefix from the token
                buff.append(int(token[4:]))
                samples.append(buff)

        df = pd.DataFrame(samples, columns=self.keys)

        return df

    def untokenize_tgt(self, tokens: list[str]) -> list[int]:
        # [6:] to remove "START-" prefix from the end of a token
        starts = [int(token[6:]) for token in tokens if token not in self.specials]

        return starts
