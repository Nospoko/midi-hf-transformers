import os
import glob
import json
import hashlib

import torch
import pandas as pd
import fortepyan as ff
from tqdm import tqdm
from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, load_dataset, concatenate_datasets

from data.midiencoder import MidiEncoder
from data.multitokencoder import MultiTokEncoder
from data.maskedmidiencoder import MaskedMidiEncoder
from data.quantizer import MidiQuantizer, MidiATQuantizer


def build_AT_translation_dataset(
    dataset: Dataset,
    dataset_cfg: DictConfig,
) -> Dataset:
    """
    Build a translation dataset using start time for quantization.
    """
    quantizer = MidiATQuantizer(
        n_duration_bins=dataset_cfg.quantization.duration,
        n_velocity_bins=dataset_cfg.quantization.velocity,
        n_start_bins=dataset_cfg.quantization.start,
        sequence_duration=dataset_cfg.sequence_duration,
    )

    sequences = []

    for it, record in tqdm(enumerate(dataset), total=len(dataset)):
        piece = ff.MidiPiece.from_huggingface(record)

        # We want to get back to the original recording easily
        piece.source |= {"base_record_id": it, "dataset_name": dataset.info.description}
        sequences += piece_to_AT_records(
            piece=piece,
            quantizer=quantizer,
            sequence_duration=dataset_cfg.sequence_duration,
            sequence_step=dataset_cfg.sequence_step,
        )

    new_dataset = Dataset.from_list(sequences)
    return new_dataset


def piece_to_AT_records(
    piece: ff.MidiPiece,
    quantizer: MidiATQuantizer,
    sequence_duration: int,
    sequence_step: int,
):
    chopped_sequences = []
    df = piece.df.copy()
    start = 0
    while start < len(df.start):
        start_time = df["start"][start]
        finish = start

        while finish < len(df["start"]) and df["start"][finish] - df["start"][start] < sequence_duration:
            finish += 1

        part = df.iloc[start:finish].copy()
        part["start"] = part["start"] - start_time
        part["end"] = part["end"] - start_time
        part = quantizer.quantize_frame(part)

        source = piece.source.copy()
        source |= {"start": start, "finish": finish}

        sequence = {
            "pitch": part.pitch.astype("int16").values.T,
            # Quantized features
            "start_bin": part.start_bin.astype("int16").values.T,
            "duration_bin": part.duration_bin.astype("int16").values.T,
            "velocity_bin": part.velocity_bin.astype("int16").values.T,
            # Ground truth
            "start": part.start.values,
            "end": part.end.values,
            "duration": part.duration.values,
            "velocity": part.velocity.values,
            "source": json.dumps(source),
        }
        if min(sequence["pitch"]) >= 21:
            chopped_sequences.append(sequence)
        if finish == len(df["start"]):
            break
        start = finish
        while df["start"][finish] - df["start"][start] > sequence_step:
            start -= 1
    return chopped_sequences


def build_translation_dataset(
    dataset: Dataset,
    dataset_cfg: DictConfig,
) -> Dataset:
    """
    Build a dataset using dstart for quantization.
    """
    # ~90s for giant midi
    quantizer = MidiQuantizer(
        n_dstart_bins=dataset_cfg.quantization.dstart,
        n_duration_bins=dataset_cfg.quantization.duration,
        n_velocity_bins=dataset_cfg.quantization.velocity,
    )

    quantized_pieces = []

    for it, record in tqdm(enumerate(dataset), total=len(dataset)):
        piece = ff.MidiPiece.from_huggingface(record)
        qpiece = quantizer.inject_quantization_features(piece)

        # We want to get back to the original recording easily
        qpiece.source |= {"base_record_id": it, "dataset_name": dataset.info.description}
        quantized_pieces.append(qpiece)

    # ~10min for giant midi
    chopped_sequences = []
    for it, piece in tqdm(enumerate(quantized_pieces), total=len(quantized_pieces)):
        chopped_sequences += quantized_piece_to_records(
            piece=piece,
            sequence_len=dataset_cfg.sequence_len,
            sequence_step=dataset_cfg.sequence_step,
        )

    new_dataset = Dataset.from_list(chopped_sequences)
    return new_dataset


def quantized_piece_to_records(piece: ff.MidiPiece, sequence_len: int, sequence_step: int):
    chopped_sequences = []
    n_samples = 1 + (piece.size - sequence_len) // sequence_step
    df = piece.df
    for jt in range(n_samples):
        start = jt * sequence_step
        finish = start + sequence_len
        part = df.iloc[start:finish]

        source = piece.source.copy()
        source |= {"start": start, "finish": finish}

        sequence = {
            "pitch": part.pitch.astype("int16").values.T,
            # Quantized features
            "dstart_bin": part.dstart_bin.astype("int16").values.T,
            "duration_bin": part.duration_bin.astype("int16").values.T,
            "velocity_bin": part.velocity_bin.astype("int16").values.T,
            # Ground truth
            "start": part.start.values,
            "end": part.end.values,
            "duration": part.duration.values,
            "velocity": part.velocity.values,
            "source": json.dumps(source),
        }
        chopped_sequences.append(sequence)
    return chopped_sequences


class MyTokenizedMidiDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_cfg: DictConfig,
        encoder: MultiTokEncoder | MidiEncoder,
    ):
        self.dataset = dataset
        self.dataset_cfg = dataset_cfg
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.dataset)

    def __rich_repr__(self):
        yield "MyTokenizedMidiDataset"
        yield "size", len(self)
        # Nicer print
        yield "cfg", OmegaConf.to_container(self.dataset_cfg)
        yield "encoder", self.encoder

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        source_tokens_ids = self.encoder.encode_src(record)
        target_tokens_ids = self.encoder.encode_tgt(record)

        source_tokens_ids, target_tokens_ids = self.add_start_token(source_tokens_ids, target_tokens_ids)

        out = {
            "source_token_ids": torch.tensor(source_tokens_ids, dtype=torch.int64),
            "target_token_ids": torch.tensor(target_tokens_ids, dtype=torch.int64),
        }
        return out

    def get_complete_record(self, idx: int) -> dict:
        # The usual token ids + everything we store
        out = self[idx] | self.dataset[idx]
        return out

    def add_start_token(self, src_token_ids: list[int], tgt_token_ids: list[int]):
        # assert that start token always has idx = 0
        cls_token_id = self.encoder.token_to_id["<CLS>"]
        src_token_ids.insert(0, cls_token_id)
        tgt_token_ids.insert(0, cls_token_id)

        return src_token_ids, tgt_token_ids


class MaskedMidiDataset(MyTokenizedMidiDataset):
    """
    Midi dataset for T5 denoising objective.
    """

    def __init__(
        self,
        dataset: Dataset,
        dataset_cfg: DictConfig,
        base_encoder: MidiEncoder | MultiTokEncoder,
        masking_probability,
    ):
        super().__init__(dataset, dataset_cfg, base_encoder)
        self.encoder = MaskedMidiEncoder(base_encoder=base_encoder, masking_probability=masking_probability)

    def __rich_repr__(self):
        yield "MaskedMidiDataset"
        yield "size", len(self)
        # Nicer print
        yield "cfg", OmegaConf.to_container(self.dataset_cfg)
        yield "encoder", self.encoder

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        source_tokens_ids, target_tokens_ids = self.encoder.encode_record(record)

        source_tokens_ids, target_tokens_ids = self.add_start_token(source_tokens_ids, target_tokens_ids)

        out = {
            "source_token_ids": torch.tensor(source_tokens_ids, dtype=torch.int64),
            "target_token_ids": torch.tensor(target_tokens_ids, dtype=torch.int64),
        }
        return out


def shard_and_build(
    dataset: Dataset,
    dataset_cfg: DictConfig,
    dataset_cache_path: str,
    num_shards: int = 2,
) -> Dataset:
    """
    Build a translation dataset using dstart quantization, from dataset with musical pieces.
    Will shard the dataset first and process shards, then concatenate.
    """
    shard_paths = []

    for it in range(num_shards):
        path = f"{dataset_cache_path}-part-{it}"

        dataset_shard = dataset.shard(num_shards=num_shards, index=it)
        print(f"Processing shard {it} of {num_shards} with {len(dataset_shard)} records.")

        processed_shard = build_translation_dataset(dataset_shard, dataset_cfg=dataset_cfg)
        processed_shard.save_to_disk(path)
        shard_paths.append(path)

    processed_dataset = concatenate_datasets([Dataset.load_from_disk(path) for path in shard_paths])

    for path in shard_paths:
        for file in glob.glob(f"{path}/*"):
            os.remove(file)
        os.rmdir(path)

    return processed_dataset


def shard_and_build_AT(
    dataset: Dataset,
    dataset_cfg: DictConfig,
    dataset_cache_path: str,
    num_shards: int = 2,
) -> Dataset:
    """
    Build a translation dataset using absolute start time quantization, from dataset with musical pieces.
    Will shard the dataset first and process shards, then concatenate.
    """
    shard_paths = []

    for it in range(num_shards):
        path = f"{dataset_cache_path}-part-{it}"

        dataset_shard = dataset.shard(num_shards=num_shards, index=it)
        print(f"Processing shard {it} of {num_shards} with {len(dataset_shard)} records.")

        processed_shard = build_AT_translation_dataset(dataset_shard, dataset_cfg=dataset_cfg)
        processed_shard.save_to_disk(path)
        shard_paths.append(path)

    processed_dataset = concatenate_datasets([Dataset.load_from_disk(path) for path in shard_paths])

    for path in shard_paths:
        for file in glob.glob(f"{path}/*"):
            os.remove(file)
        os.rmdir(path)

    return processed_dataset


def load_cache_dataset(
    dataset_cfg: DictConfig,
    dataset_name: str,
    split: str = "train",
    force_build: bool = False,
) -> Dataset:
    # Prepare caching hash
    config_hash = hashlib.sha256()
    config_string = json.dumps(OmegaConf.to_container(dataset_cfg)) + dataset_name + split
    config_hash.update(config_string.encode())
    config_hash = config_hash.hexdigest()

    dataset_cache_path = f"tmp/datasets/{config_hash}"

    if not force_build:
        try:
            translation_dataset = Dataset.load_from_disk(dataset_cache_path)
            return translation_dataset
        except Exception as e:
            print("Failed loading cached dataset:", e)

    print("Building translation dataset from", dataset_name, split)
    midi_dataset = load_dataset(dataset_name, split=split)

    # using description to store dataset_name allows sharding to work properly
    midi_dataset.info.description = dataset_name

    # hardcoded maximum shard size as 5000
    num_shards = len(midi_dataset) // 5000 + 1
    if "dstart" in dataset_cfg.quantization:
        translation_dataset = shard_and_build(
            dataset=midi_dataset,
            dataset_cfg=dataset_cfg,
            num_shards=num_shards,
            dataset_cache_path=dataset_cache_path,
        )
    else:
        translation_dataset = shard_and_build_AT(
            dataset=midi_dataset,
            dataset_cfg=dataset_cfg,
            num_shards=num_shards,
            dataset_cache_path=dataset_cache_path,
        )
    translation_dataset.save_to_disk(dataset_cache_path)
    # load dataset again to update cache_files attribute
    translation_dataset = Dataset.load_from_disk(dataset_cache_path)

    return translation_dataset


def main():
    dataset_name = "roszcz/maestro-v1-sustain"
    dataset_cfg = {
        "sequence_duration": 5,
        "sequence_step": 10,
        "quantization": {
            "duration": 3,
            "velocity": 3,
            # 650 start bins sound nice :)
            "start": 20,
        },
    }
    cfg = OmegaConf.create(dataset_cfg)
    dataset = load_cache_dataset(cfg, dataset_name, split="test")

    quantizer = MidiATQuantizer(
        n_duration_bins=cfg.quantization.duration,
        n_velocity_bins=cfg.quantization.velocity,
        n_start_bins=cfg.quantization.start,
        sequence_duration=cfg.sequence_duration,
    )

    lengths = [len(record["pitch"]) for record in dataset]
    lengths = sorted(lengths)
    maximum = max(lengths)
    print(maximum)
    plt.plot(lengths)
    plt.ylabel("length")
    plt.xlabel("idx")
    plt.show()

    record = pd.DataFrame(dataset[0])
    record = quantizer.apply_quantization(record)
    piece = ff.MidiPiece(record)
    print(piece.df)
    # ff.view.make_piano_roll_video(piece, "test.mp4")

    from data.multitokencoder import MultiVelocityEncoder

    # this is for testing and debugging btw
    base_encoder = MultiVelocityEncoder(cfg.quantization, time_quantization_method="start")
    test_dataset = MaskedMidiDataset(
        dataset=dataset,
        dataset_cfg=cfg,
        base_encoder=base_encoder,
        masking_probability=0.3,
    )
    record = test_dataset[90]

    df = test_dataset.encoder.decode(record["source_token_ids"], record["target_token_ids"])
    print(df)


if __name__ == "__main__":
    main()
