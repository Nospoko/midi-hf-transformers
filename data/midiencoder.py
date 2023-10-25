import itertools

import pandas as pd
from omegaconf import DictConfig


class MidiEncoder:
    def __init__(self):
        self.token_to_id = None
        self.vocab = None

    def tokenize_src(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *tokenize* implementation")

    def tokenize_tgt(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *tokenize* implementation")

    def untokenize_src(self, tokens: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Your encoder needs *untokenize* implementation")

    def untokenize_tgt(self, tokens: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Your encoder needs *untokenize* implementation")

    def decode_src(self, token_ids: list[int]) -> pd.DataFrame:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        df = self.untokenize_src(tokens)

        return df

    def decode_tgt(self, token_ids: list[int]) -> pd.DataFrame:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        df = self.untokenize_tgt(tokens)

        return df

    def encode_src(self, record: dict) -> list[int]:
        tokens = self.tokenize_src(record)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids

    def encode_tgt(self, record: dict) -> list[int]:
        tokens = self.tokenize_tgt(record)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids


class QuantizedMidiEncoder(MidiEncoder):
    def __init__(self, quantization_cfg: DictConfig, time_quantization_method: str):
        super().__init__()
        self.quantization_cfg = quantization_cfg
        self.time_quantization_key = time_quantization_method + "_bin"
        self.keys = ["pitch", time_quantization_method, "duration_bin", "velocity_bin"]
        self.specials = ["<CLS>"]

        self.vocab = list(self.specials)

        # add midi tokens to vocab
        self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "QuantizedMidiEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self):
        src_iterators_product = itertools.product(
            # Always include 88 pitches
            range(21, 109),
            range(self.quantization_cfg[self.time_quantization_key]),
            range(self.quantization_cfg.duration),
            range(self.quantization_cfg.velocity),
        )

        for pitch, dstart, duration, velocity in src_iterators_product:
            key = f"{pitch}-{dstart}-{duration}-{velocity}"
            self.vocab.append(key)

    def tokenize_src(self, record: dict) -> list[str]:
        tokens = []
        n_samples = len(record["pitch"])
        for idx in range(n_samples):
            token = "-".join([f"{record[key][idx]:0.0f}" for key in self.keys])
            tokens.append(token)

        return tokens

    def untokenize_src(self, tokens: list[str]) -> pd.DataFrame:
        samples = []
        for token in tokens:
            if token in self.specials:
                continue

            values_txt = token.split("-")
            values = [eval(txt) for txt in values_txt]
            samples.append(values)

        df = pd.DataFrame(samples, columns=self.keys)

        return df


class VelocityEncoder(MidiEncoder):
    def __init__(self, quantization_cfg: DictConfig, time_quantization_method: str):
        super().__init__()
        self.key = "velocity"
        self.specials = ["<CLS>", "<PAD>"]
        self._src_encoder = QuantizedMidiEncoder(quantization_cfg, time_quantization_method)

        # take vocab from src_encoder
        self.vocab = self._src_encoder.vocab
        # add velocity tokens
        self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "VelocityEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self):
        self.vocab += [str(possible_velocity) for possible_velocity in range(128)]

    def tokenize_src(self, record: dict) -> list[str]:
        return self._src_encoder.tokenize_src(record)

    def tokenize_tgt(self, record: dict) -> list[str]:
        tokens = [str(velocity) for velocity in record["velocity"]]
        return tokens

    def untokenize_tgt(self, tokens: list[str]) -> list[int]:
        velocities = [int(token) for token in tokens if token not in self.specials]

        return velocities

    def untokenize_src(self, tokens: list[str]) -> pd.DataFrame:
        return self._src_encoder.untokenize_src(tokens)
