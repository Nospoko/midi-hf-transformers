import os

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
    # 88 piano keys
    size = 88
    values = [cfg.dataset.quantization[key] for key in cfg.dataset.quantization]
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


def calculate_average_distance(out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    labels = out.argmax(1).to(float)
    # average distance between label and target
    return torch.dist(labels, tgt.to(float), p=1) / len(labels)
