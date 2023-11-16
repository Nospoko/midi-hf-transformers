import os
from math import sqrt

import torch
import pretty_midi
import fortepyan as ff
import matplotlib.pyplot as plt
from fortepyan import MidiPiece
from omegaconf import DictConfig
import fortepyan.audio.render as render_audio


def piece_av_files(piece: MidiPiece, save_base: str) -> dict:
    # fixed by Tomek
    mp3_path = save_base + ".mp3"

    if not os.path.exists(mp3_path):
        render_audio.midi_to_mp3(piece.to_midi(), mp3_path)

    pianoroll_path = save_base + ".png"

    if not os.path.exists(pianoroll_path):
        if "mask" in piece.df.keys():
            ff.view.draw_dual_pianoroll(piece)
        else:
            ff.view.draw_pianoroll_with_velocities(piece)
        plt.tight_layout()
        plt.savefig(pianoroll_path)
        plt.clf()

    midi_path = save_base + ".mid"
    if not os.path.exists(midi_path):
        # Add a silent event to make sure the final notes
        # have time to ring out
        midi = piece.to_midi()
        end_time = midi.get_end_time() + 0.2
        pedal_off = pretty_midi.ControlChange(64, 0, end_time)
        midi.instruments[0].control_changes.append(pedal_off)
        midi.write(midi_path)

    paths = {
        "mp3_path": mp3_path,
        "midi_path": midi_path,
        "pianoroll_path": pianoroll_path,
    }
    return paths


def vocab_size(cfg: DictConfig):
    # 88 pitches
    size: int = 88

    if cfg.target == "denoise" or ("finetune" in cfg.train and cfg.train.finetune):
        if cfg.tokens_per_note == "single":
            # product size
            size = size * cfg.dataset.quantization[cfg.time_quantization_method]
            size = size * cfg.dataset.quantization.duration
            size = size * cfg.dataset.quantization.velocity
            # velocity tokens
            size += 128
            size += cfg.time_bins * 10
            size += 103
            return size
        size += 103  # 100 sentinel tokens, 1 mask token and 2 special tokens
        size += 128  # velocity tokens
        size += cfg.time_bins * 10  # start * duration or dstart * duration
        size += cfg.time_bins

        return size

    if cfg.tokens_per_note == "single":
        # product size
        size = size * cfg.dataset.quantization[cfg.time_quantization_method]
        size = size * cfg.dataset.quantization.duration
        size = size * cfg.dataset.quantization.velocity
        # velocity tokens
        size += 128
        # special tokens
        size += 2
        return size

    values = [cfg.dataset.quantization[key] for key in cfg.dataset.quantization]

    # 2 special tokens - <CLS> and <PAD>
    size += 2
    # time tokens
    size += values[0] * values[1]
    # velocity tokens
    if cfg.target == "velocity":
        size += 128
        return size
    size += values[2]

    if cfg.target == "start":
        size += cfg.start_bins

    return size


def calculate_average_distance(out: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    labels = out.argmax(1).to(float)
    tgt[tgt == -100] = pad_idx
    tgt = tgt[tgt != pad_idx]
    # make labels fixed length same as target
    labels = labels[: len(tgt)]

    # remove special tokens
    labels[tgt == pad_idx] = pad_idx

    # average distance between label and target
    return torch.dist(labels, tgt.to(float), p=1) / len(labels)


def learning_rate_schedule(step: int, warmup: int):
    return 1 / sqrt(max(step, warmup))


def debug_lr_schedule(step: int):
    return 1e-4 * 10 ** ((step // 10) / 20)
