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

        self.specials = ["<CLS>", "<pad>"]

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
            self.vocab.append(f"{pitch}p")
        for start, duration in time_tokens_product:
            self.vocab.append(f"{start:0.0f}-{duration:0.0f}")
        # in hf T5 and BART we will use common vocabulary for tgt and src tokens ¯\_(ツ)_/¯
        # luckily, MultiTok is ideal for this
        # (UNLESS we will manually swap a final layer (somehow))
        # TODO: find out how sharing tokens effects model performance. (1v, 2v, 3v for 3 bins and also for 128 bins)
        for velocity in range(128):
            self.vocab.append(f"{velocity:0.0f}v")

    def tokenize_src(self, record: dict) -> list[str]:
        tokens = []
        n_samples = len(record[self.keys[0]])
        for idx in range(n_samples):
            # append p and v so that there are no duplicate tokens
            pitch_token = f"{record[self.keys[0]][idx]:0.0f}p"
            time_token = f"{record[self.keys[1]][idx]}-{record[self.keys[2]][idx]:0.0f}"
            velocity_token = f"{record[self.keys[3]][idx]:0.0f}v"
            tokens += [pitch_token, time_token, velocity_token]

        return tokens

    def tokenize_tgt(self, record: dict) -> list[str]:
        tokens = [f"{velocity:0.0f}v" for velocity in record["velocity"]]
        return tokens

    def untokenize_src(self, tokens: list[str]) -> pd.DataFrame:
        samples = []
        buff = []
        if tokens[0] == "<CLS>":
            # get rid of cls token
            tokens = tokens[1:]

        for idx, token in enumerate(tokens):
            if token == "<pad>":
                break
            if idx % 3 == 0:
                # pitch token
                buff.clear()
                # [:-1] to clear p letter from a token
                buff.append(eval(token[:-1]))
            elif idx % 3 == 1:
                # time token
                buff.append(eval(txt) for txt in token.split("-"))
            else:
                # velocity token
                # [:-1] to clear v letter from the token
                buff.append(eval(token[:-1]))
                samples.append(buff)

        df = pd.DataFrame(samples, columns=self.keys)

        return df

    def untokenize_tgt(self, tokens: list[str]) -> list[int]:
        # [:-1] to remove v letter from the end of a token
        velocities = [int(token[:-1]) for token in tokens if token not in self.specials]

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

        self.specials = ["<CLS>", "<pad>"]

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
            self.vocab.append(f"{pitch}p")
        for start, duration in time_tokens_product:
            self.vocab.append(f"{start:0.0f}-{duration:0.0f}")
        for velocity in range(self.quantization_cfg.velocity):
            self.vocab.append(f"{velocity:0.0f}v")
        for start in range(self.tgt_bins):
            self.vocab.append(f"{start}s")

    def tokenize_src(self, record: dict) -> list[str]:
        tokens = []
        n_samples = len(record[self.keys[0]])
        for idx in range(n_samples):
            # append p and v so that there are no duplicate tokens
            pitch_token = f"{record[self.keys[0]][idx]:0.0f}p"
            time_token = f"{record[self.keys[1]][idx]}-{record[self.keys[2]][idx]:0.0f}"
            velocity_token = f"{record[self.keys[3]][idx]:0.0f}v"
            tokens += [pitch_token, time_token, velocity_token]

        return tokens

    def quantize_start(self, starts: pd.Series):
        bin_edges = np.linspace(start=0, stop=len(starts), num=self.tgt_bins)

        quantized_starts = np.digitize(starts, bin_edges)
        return quantized_starts

    def tokenize_tgt(self, record: dict) -> list[str]:
        quantized_starts = self.quantize_start(record["start"]) - 1
        tokens = [f"{start_bin:0.0f}s" for start_bin in quantized_starts]
        return tokens

    def untokenize_src(self, tokens: list[str]) -> pd.DataFrame:
        samples = []
        buff = []
        if tokens[0] == "<CLS>":
            # get rid of cls token
            tokens = tokens[1:]

        for idx, token in enumerate(tokens):
            if token == "<pad>":
                break
            if idx % 3 == 0:
                # pitch token
                buff.clear()
                # [:-1] to clear p letter from a token
                buff.append(eval(token[:-1]))
            elif idx % 3 == 1:
                # time token
                buff.append(eval(txt) for txt in token.split("-"))
            else:
                # velocity token
                # [:-1] to clear v letter from the token
                buff.append(eval(token[:-1]))
                samples.append(buff)

        df = pd.DataFrame(samples, columns=self.keys)

        return df

    def untokenize_tgt(self, tokens: list[str]) -> list[int]:
        # [:-1] to remove v letter from the end of a token
        starts = [int(token[:-1]) for token in tokens if token not in self.specials]

        return starts
