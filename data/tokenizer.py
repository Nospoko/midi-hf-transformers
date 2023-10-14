import itertools

import pandas as pd
from omegaconf import DictConfig


class MidiEncoder:
    def __init__(self):
        self.token_to_id = None

    def tokenize_src(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *tokenize_src* implementation")

    def untokenize_src(self, tokens: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Your encoder needs *untokenize_src* implementation")

    def tokenize_tgt(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *tokenize_tgt* implementation")

    def untokenize_tgt(self, tokens: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Your encoder needs *untokenize_tgt* implementation")

    def decode(self, token_ids: list[int]) -> pd.DataFrame:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        df = self.untokenize(tokens)

        return df

    def encode_src(self, record: dict) -> list[int]:
        tokens = self.tokenize_src(record)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids

    def encode_tgt(self, record: dict) -> list[int]:
        tokens = self.tokenize_tgt(record)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids


class MultiTokEncoder(MidiEncoder):
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
        yield "QuantizedMidiEncoder"
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
        # in hf T5 and BERT we will use common vocabulary for tgt and src tokens ¯\_(ツ)_/¯
        # luckily, MultiTok is ideal for this
        # (UNLESS we will manually swap a final layer (somehow))
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
