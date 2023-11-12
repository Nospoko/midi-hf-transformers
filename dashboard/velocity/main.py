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

from data.midiencoder import VelocityEncoder
from utils import vocab_size, piece_av_files
from data.maskedmidiencoder import MaskedMidiEncoder
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset
from data.multitokencoder import MultiMidiEncoder, MultiVelocityEncoder

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

    if train_cfg.train.finetune:
        tokenizer = MultiMidiEncoder(
            quantization_cfg=train_cfg.dataset.quantization,
            time_quantization_method=train_cfg.time_quantization_method,
        )
        pretraining_tokenizer = MaskedMidiEncoder(
            base_encoder=tokenizer,
        )
        # use the same token ids as used during pre-training
        tokenizer.vocab = pretraining_tokenizer.vocab
        tokenizer.token_to_id = pretraining_tokenizer.token_to_id
        tokenizer.specials = pretraining_tokenizer.specials
    elif train_cfg.tokens_per_note == "multiple":
        tokenizer = MultiVelocityEncoder(
            quantization_cfg=train_cfg.dataset.quantization,
            time_quantization_method=train_cfg.time_quantization_method,
        )
    else:
        tokenizer = VelocityEncoder(
            quantization_cfg=train_cfg.dataset.quantization,
            time_quantization_method=train_cfg.time_quantization_method,
        )
    cls_token_id = tokenizer.token_to_id["<CLS>"]
    pad_token_id = tokenizer.token_to_id["<PAD>"]
    config = T5Config(
        vocab_size=vocab_size(train_cfg),
        decoder_start_token_id=cls_token_id,
        eos_token_id=pad_token_id,
        pad_token_id=pad_token_id,
        use_cache=False,
    )

    model = T5ForConditionalGeneration(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(DEVICE)

    n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    st.markdown(f"Model parameters: {n_parameters:.3f}M")

    # Folder to render audio and video
    model_dir = f"tmp/dashboard/{train_cfg.run_name}"

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if mode == "Sequence predictions":
        model_predictions_review(
            model=model,
            train_cfg=train_cfg,
            model_dir=model_dir,
            tokenizer=tokenizer,
        )


def model_predictions_review(
    model: nn.Module,
    train_cfg: DictConfig,
    model_dir: str,
    tokenizer: MultiVelocityEncoder | MultiMidiEncoder | VelocityEncoder,
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

    dataset = MyTokenizedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=train_cfg.dataset,
        encoder=tokenizer,
    )

    n_samples = 5
    np.random.seed(random_seed)
    idxs = np.random.randint(len(dataset), size=n_samples)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Predicted")

    # predict velocities and get src, tgt and model output
    print("Making predictions ...")
    for record_id in idxs:
        # Numpy to int :(
        record = dataset.get_complete_record(int(record_id))
        record_source = json.loads(record["source"])
        src_token_ids = record["source_token_ids"]

        if train_cfg.tokens_per_note == "multiple":
            max_length = len(src_token_ids) // 3 + 1
        else:
            max_length = len(src_token_ids)
        generated_velocity = model.generate(src_token_ids.unsqueeze(0), max_length=max_length)
        generated_velocity = generated_velocity.squeeze(0)
        decoded_velocity = tokenizer.decode_tgt(generated_velocity)

        # TODO start here
        # Reconstruct the sequence as recorded
        midi_columns = ["pitch", "start", "end", "duration", "velocity"]
        true_notes = pd.DataFrame({c: record[c] for c in midi_columns})
        true_piece = MidiPiece(df=true_notes, source=record_source)
        true_piece.time_shift(-true_piece.df.start.min())

        pred_piece_df = true_piece.df.copy()

        # change untokenized velocities to model predictions
        pred_piece_df["velocity"] = decoded_velocity
        pred_piece_df["velocity"] = pred_piece_df["velocity"].fillna(0)

        # create quantized piece with predicted velocities
        pred_piece = MidiPiece(pred_piece_df)

        pred_piece.source = true_piece.source.copy()

        # create files
        true_save_base = os.path.join(model_dir, f"true_{record_id}")
        true_piece_paths = piece_av_files(piece=true_piece, save_base=true_save_base)

        predicted_save_base = os.path.join(model_dir, f"predicted_{record_id}")
        predicted_paths = piece_av_files(piece=pred_piece, save_base=predicted_save_base)

        # create a dashboard
        st.json(record_source)
        cols = st.columns(2)
        source_tokens: list[str] = [dataset.encoder.vocab[idx] for idx in src_token_ids]
        tgt_tokens: list[str] = [dataset.encoder.vocab[idx] for idx in record["target_token_ids"]]
        generated_tokens: list[str] = [dataset.encoder.vocab[idx] for idx in generated_velocity]
        with cols[0]:
            # Unchanged
            st.image(true_piece_paths["pianoroll_path"])
            st.audio(true_piece_paths["mp3_path"])
            st.markdown("**Source tokens:**")
            st.markdown(source_tokens)
            st.markdown("**Target tokens:**")
            st.markdown(tgt_tokens)

        with cols[1]:
            # Predicted
            st.image(predicted_paths["pianoroll_path"])
            st.audio(predicted_paths["mp3_path"])
            st.markdown("**Predicted tokens:**")
            st.markdown(generated_tokens)


if __name__ == "__main__":
    main()
