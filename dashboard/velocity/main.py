import os
import glob
import json

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import streamlit as st
from fortepyan import MidiPiece
from omegaconf import OmegaConf, DictConfig
from transformers import T5Config, T5ForConditionalGeneration

from data.quantizer import MidiATQuantizer
from utils import vocab_size, piece_av_files
from data.tokenizer import MultiVelocityEncoder
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset

# Set the layout of the Streamlit page
st.set_page_config(layout="wide", page_title="Velocity Transformer", page_icon=":musical_keyboard")

with st.sidebar:
    devices = ["cpu"] + [f"cuda:{it}" for it in range(torch.cuda.device_count())]
    DEVICE = st.selectbox(label="Processing device", options=devices)


@torch.no_grad()
def main():
    with st.sidebar:
        dashboards = [
            "Sequence predictions",
        ]
        mode = st.selectbox(label="Display", options=dashboards)

    with st.sidebar:
        # Show available checkpoints
        options = glob.glob("checkpoints/velocity/*.pt")
        options.sort()
        checkpoint_path = st.selectbox(label="model", options=options)
        st.markdown("Selected checkpoint:")
        st.markdown(checkpoint_path)

    # Load:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # - original config
    train_cfg = OmegaConf.create(checkpoint["cfg"])
    train_cfg.device = DEVICE

    # - - for dataset
    dataset_params = OmegaConf.to_container(train_cfg.dataset)
    st.markdown("Dataset config:")
    st.json(dataset_params, expanded=True)

    config = T5Config(
        vocab_size=vocab_size(train_cfg),
        decoder_start_token_id=0,
        use_cache=False,
    )

    model = T5ForConditionalGeneration(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(DEVICE)

    quantizer = MidiATQuantizer(
        n_duration_bins=train_cfg.dataset.quantization.duration,
        n_velocity_bins=train_cfg.dataset.quantization.velocity,
        n_start_bins=train_cfg.dataset.quantization.start,
        sequence_duration=train_cfg.dataset.sequence_duration,
    )

    n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    st.markdown(f"Model parameters: {n_parameters:.3f}M")

    # Folder to render audio and video
    model_dir = f"tmp/dashboard/{train_cfg.run_name}"

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if mode == "Sequence predictions":
        model_predictions_review(
            model=model,
            quantizer=quantizer,
            train_cfg=train_cfg,
            model_dir=model_dir,
        )


def model_predictions_review(
    model: nn.Module,
    quantizer: MidiATQuantizer,
    train_cfg: DictConfig,
    model_dir: str,
):
    # load checkpoint, force dashboard device
    dataset_cfg = train_cfg.dataset
    dataset_name = st.text_input(label="dataset", value=train_cfg.dataset_name)
    split = st.text_input(label="split", value="test")

    random_seed = st.selectbox(label="random seed", options=range(20))

    # load translation dataset and create MyTokenizedMidiDataset
    val_translation_dataset = load_cache_dataset(
        dataset_cfg=dataset_cfg,
        dataset_name=dataset_name,
        split=split,
    )
    keys = ["pitch", "start_bin", "duration_bin", "velocity_bin"]
    tokenizer = MultiVelocityEncoder(
        quantization_cfg=train_cfg.dataset.quantization,
        keys=keys,
    )

    dataset = MyTokenizedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=train_cfg.dataset,
        encoder=tokenizer,
    )

    n_samples = 5
    np.random.seed(random_seed)
    idxs = np.random.randint(len(dataset), size=n_samples)

    cols = st.columns(3)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Q. velocity")
    with cols[2]:
        st.markdown("### Predicted")

    # predict velocities and get src, tgt and model output
    print("Making predictions ...")
    for record_id in idxs:
        # Numpy to int :(
        record = dataset.get_complete_record(int(record_id))
        record_source = json.loads(record["source"])
        src_token_ids = record["source_token_ids"]

        max_length = len(src_token_ids) // 3 + 1
        generated_velocity = model.generate(src_token_ids.unsqueeze(0), max_length=max_length)
        generated_velocity = generated_velocity.squeeze(0)
        generated_velocity = tokenizer.decode_tgt(generated_velocity)
        print(generated_velocity)

        # Just pitches and quantization n_bins of the source
        src_tokens = [dataset.encoder.vocab[token_id] for token_id in src_token_ids]
        source_df = dataset.encoder.untokenize_src(src_tokens)

        quantized_notes = quantizer.apply_quantization(source_df)
        quantized_piece = MidiPiece(quantized_notes)
        quantized_piece.time_shift(-quantized_piece.df.start.min())

        # TODO start here
        # Reconstruct the sequence as recorded
        midi_columns = ["pitch", "start", "end", "duration", "velocity"]
        true_notes = pd.DataFrame({c: record[c] for c in midi_columns})
        true_piece = MidiPiece(df=true_notes, source=record_source)
        true_piece.time_shift(-true_piece.df.start.min())

        pred_piece_df = true_piece.df.copy()
        quantized_vel_df = true_piece.df.copy()

        # change untokenized velocities to model predictions
        pred_piece_df["velocity"] = generated_velocity
        pred_piece_df["velocity"] = pred_piece_df["velocity"].fillna(0)

        quantized_vel_df["velocity"] = quantized_piece.df["velocity"].copy()

        # create quantized piece with predicted velocities
        pred_piece = MidiPiece(pred_piece_df)
        quantized_vel_piece = MidiPiece(quantized_vel_df)

        pred_piece.source = true_piece.source.copy()
        quantized_vel_piece.source = true_piece.source.copy()

        # create files
        true_save_base = os.path.join(model_dir, f"true_{record_id}")
        true_piece_paths = piece_av_files(piece=true_piece, save_base=true_save_base)

        qv_save_base = os.path.join(model_dir, f"qv_{record_id}")
        qv_paths = piece_av_files(piece=quantized_vel_piece, save_base=qv_save_base)

        predicted_save_base = os.path.join(model_dir, f"predicted_{record_id}")
        predicted_paths = piece_av_files(piece=pred_piece, save_base=predicted_save_base)

        # create a dashboard
        st.json(record_source)
        cols = st.columns(3)
        with cols[0]:
            # Unchanged
            st.image(true_piece_paths["pianoroll_path"])
            st.audio(true_piece_paths["mp3_path"])

        with cols[1]:
            # Q.velocity ?
            st.image(qv_paths["pianoroll_path"])
            st.audio(qv_paths["mp3_path"])

        with cols[2]:
            # Predicted
            st.image(predicted_paths["pianoroll_path"])
            st.audio(predicted_paths["mp3_path"])


if __name__ == "__main__":
    main()
