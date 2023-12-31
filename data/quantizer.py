import yaml
import numpy as np
import pandas as pd
from fortepyan import MidiPiece
from hydra.utils import to_absolute_path


class MidiQuantizer:
    def __init__(
        self,
        n_dstart_bins: int = 3,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
    ):
        self.n_dstart_bins = n_dstart_bins
        self.n_duration_bins = n_duration_bins
        self.n_velocity_bins = n_velocity_bins
        self._build()

    def __rich_repr__(self):
        yield "MidiQuantizer"
        yield "n_dstart_bins", self.n_dstart_bins
        yield "n_duration_bins", self.n_duration_bins
        yield "n_velocity_bins", self.n_velocity_bins

    def _build(self):
        self._load_bin_edges()
        self._build_dstart_decoder()
        self._build_duration_decoder()
        self._build_velocity_decoder()

    def _load_bin_edges(self):
        # Hydra changes paths, this finds it back
        artifacts_path = to_absolute_path("artifacts/bin_edges.yaml")
        with open(artifacts_path, "r") as f:
            bin_edges = yaml.safe_load(f)

        self.dstart_bin_edges = bin_edges["dstart"][self.n_dstart_bins]
        self.duration_bin_edges = bin_edges["duration"][self.n_duration_bins]
        self.velocity_bin_edges = bin_edges["velocity"][self.n_velocity_bins]

    def inject_quantization_features(self, piece: MidiPiece) -> MidiPiece:
        # Try not to overwrite anything
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Make the quantization
        df = self.quantize_frame(df)
        out = MidiPiece(df=df, source=source)
        return out

    def quantize_piece(self, piece: MidiPiece) -> MidiPiece:
        # Try not to overwrite anything
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Make the quantization
        df = self.quantize_frame(df)
        df = self.apply_quantization(df)
        out = MidiPiece(df=df, source=source)
        return out

    def quantize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df["next_start"] = df.start.shift(-1)
        df["dstart"] = df.next_start - df.start
        df["dstart_bin"] = np.digitize(df.dstart.fillna(0), self.dstart_bin_edges) - 1
        df["duration_bin"] = np.digitize(df.duration, self.duration_bin_edges) - 1
        df["velocity_bin"] = np.digitize(df.velocity, self.velocity_bin_edges) - 1

        return df

    def quantize_velocity(self, velocity: np.array) -> np.array:
        velocity_bins = np.digitize(velocity, self.velocity_bin_edges) - 1
        quantized_velocity = np.array([self.bin_to_velocity[v_bin] for v_bin in velocity_bins])
        return quantized_velocity

    def apply_quantization(self, df: pd.DataFrame) -> pd.DataFrame:
        df["quant_dstart"] = df.dstart_bin.map(lambda it: self.bin_to_dstart[it])
        df["quant_duration"] = df.duration_bin.map(lambda it: self.bin_to_duration[it])
        df["start"] = df.quant_dstart.cumsum().shift(1).fillna(0)
        df["end"] = df.start + df.quant_duration
        df["duration"] = df.quant_duration
        df["velocity"] = df.velocity_bin.map(lambda it: self.bin_to_velocity[it])
        return df

    def _build_duration_decoder(self):
        self.bin_to_duration = []
        for it in range(1, len(self.duration_bin_edges)):
            duration = (self.duration_bin_edges[it - 1] + self.duration_bin_edges[it]) / 2
            self.bin_to_duration.append(duration)

        last_duration = 2 * self.duration_bin_edges[-1]
        self.bin_to_duration.append(last_duration)

    def _build_dstart_decoder(self):
        self.bin_to_dstart = []
        for it in range(1, len(self.dstart_bin_edges)):
            dstart = (self.dstart_bin_edges[it - 1] + self.dstart_bin_edges[it]) / 2
            self.bin_to_dstart.append(dstart)

        last_dstart = 2 * self.dstart_bin_edges[-1]
        self.bin_to_dstart.append(last_dstart)

    def _build_velocity_decoder(self):
        # For velocity the first bin is not going to be
        # evenly populated, skewing towards to higher values
        # (who plays with velocity 0?)
        self.bin_to_velocity = [int(0.8 * self.velocity_bin_edges[1])]

        for it in range(2, len(self.velocity_bin_edges)):
            dstart = (self.velocity_bin_edges[it - 1] + self.velocity_bin_edges[it]) / 2
            self.bin_to_velocity.append(int(dstart))

    def make_vocab(self) -> list[str]:
        vocab = []
        for it, pitch in enumerate(range(21, 109)):
            for jt in range(self.n_duration_bins):
                for kt in range(self.n_dstart_bins):
                    for vt in range(self.n_velocity_bins):
                        key = f"{kt}_{jt}_{vt}_{pitch}"
                        vocab.append(key)

        return vocab


class MidiATQuantizer:
    def __init__(
        self,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
        n_start_bins: int = 625,
        sequence_duration: float = 20.0,
    ):
        self.n_velocity_bins = n_velocity_bins
        self.n_duration_bins = n_duration_bins
        self.n_start_bins = n_start_bins
        self.sequence_duration = sequence_duration
        self._build()

    def __rich_repr__(self):
        yield "MidiATQuantizer"
        yield "n_duration_bins", self.n_duration_bins
        yield "n_velocity_bins", self.n_velocity_bins
        yield "n_start_bins", self.n_start_bins
        yield "sequence_duration", self.sequence_duration

    def _build(self):
        self._load_bin_edges()
        self._build_start_decoder()
        self._build_duration_decoder()
        self._build_velocity_decoder()

    def _load_bin_edges(self):
        # Hydra changes paths, this finds it back
        artifacts_path = to_absolute_path("artifacts/bin_edges.yaml")
        with open(artifacts_path, "r") as f:
            bin_edges = yaml.safe_load(f)

        self.duration_bin_edges = bin_edges["duration"][self.n_duration_bins]
        self.velocity_bin_edges = bin_edges["velocity"][self.n_velocity_bins]
        self.start_bin_edges = np.linspace(start=0.0, stop=self.sequence_duration, num=self.n_start_bins)

    def _build_start_decoder(self):
        self.bin_to_start = []
        for it in range(1, len(self.start_bin_edges)):
            start = (self.start_bin_edges[it - 1] + self.start_bin_edges[it]) / 2
            self.bin_to_start.append(start)

        last_start = 2 * self.start_bin_edges[-1]
        self.bin_to_start.append(last_start)

    def _build_duration_decoder(self):
        self.bin_to_duration = []
        for it in range(1, len(self.duration_bin_edges)):
            duration = (self.duration_bin_edges[it - 1] + self.duration_bin_edges[it]) / 2
            self.bin_to_duration.append(duration)

        last_duration = 2 * self.duration_bin_edges[-1]
        self.bin_to_duration.append(last_duration)

    def _build_velocity_decoder(self):
        # For velocity the first bin is not going to be
        # evenly populated, skewing towards to higher values
        # (who plays with velocity 0?)
        self.bin_to_velocity = [int(0.8 * self.velocity_bin_edges[1])]

        for it in range(2, len(self.velocity_bin_edges)):
            velocity = (self.velocity_bin_edges[it - 1] + self.velocity_bin_edges[it]) / 2
            self.bin_to_velocity.append(int(velocity))

    def inject_quantization_features(self, piece: MidiPiece) -> MidiPiece:
        # Try not to overwrite anything
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Make the quantization
        df = self.quantize_frame(df)
        out = MidiPiece(df=df, source=source)
        return out

    def quantize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df["start_bin"] = np.digitize(df.start, self.start_bin_edges) - 1
        df["duration_bin"] = np.digitize(df.duration, self.duration_bin_edges) - 1
        df["velocity_bin"] = np.digitize(df.velocity, self.velocity_bin_edges) - 1

        return df

    def apply_quantization(self, df: pd.DataFrame) -> pd.DataFrame:
        df["start"] = df.start_bin.map(lambda it: self.bin_to_start[it])
        df["duration"] = df.duration_bin.map(lambda it: self.bin_to_duration[it])
        df["end"] = df.start + df.duration
        df["velocity"] = df.velocity_bin.map(lambda it: self.bin_to_velocity[it])
        return df

    def quantize_piece(self, piece: MidiPiece) -> MidiPiece:
        # Try not to overwrite anything
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Make the quantization
        df = self.quantize_frame(df)
        df = self.apply_quantization(df)
        out = MidiPiece(df=df, source=source)
        return out
