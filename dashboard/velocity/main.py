import os
import glob
import json

import torch
import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
from fortepyan import MidiPiece
from omegaconf import OmegaConf, DictConfig
from streamlit_pianoroll import from_fortepyan
from transformers import T5Config, T5ForConditionalGeneration

from utils import vocab_size
from data.midiencoder import VelocityEncoder
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
        options.sort(reverse=True)
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

    # Folder to render audio and video
    model_dir = f"tmp/dashboard/{train_cfg.run_name}"

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if mode == "Sequence predictions":
        model_predictions_review(
            checkpoint=checkpoint,
            train_cfg=train_cfg,
        )


def model_predictions_review(
    checkpoint: dict,
    train_cfg: DictConfig,
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

    if "finetune" in train_cfg.train and train_cfg.train.finetune:
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

    dataset = MyTokenizedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=train_cfg.dataset,
        encoder=tokenizer,
    )
    start_token_id: int = dataset.encoder.token_to_id["<CLS>"]
    pad_token_id: int = dataset.encoder.token_to_id["<PAD>"]
    config = T5Config(
        vocab_size=vocab_size(train_cfg),
        decoder_start_token_id=start_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=pad_token_id,
        use_cache=False,
        d_model=train_cfg.model.d_model,
        d_kv=train_cfg.model.d_kv,
        d_ff=train_cfg.model.d_ff,
        num_layers=train_cfg.model.num_layers,
        num_heads=train_cfg.model.num_heads,
    )

    model = T5ForConditionalGeneration(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(DEVICE)

    n_parameters: float = sum(p.numel() for p in model.parameters()) / 1e6
    st.markdown(f"Model parameters: {n_parameters:.3f}M")

    start_token_id: int = dataset.encoder.token_to_id["<CLS>"]
    pad_token_id: int = dataset.encoder.token_to_id["<PAD>"]
    config = T5Config(
        vocab_size=vocab_size(train_cfg),
        decoder_start_token_id=start_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=pad_token_id,
        use_cache=False,
        d_model=train_cfg.model.d_model,
        d_kv=train_cfg.model.d_kv,
        d_ff=train_cfg.model.d_ff,
        num_layers=train_cfg.model.num_layers,
        num_heads=train_cfg.model.num_heads,
    )

    model = T5ForConditionalGeneration(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(DEVICE)

    n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    st.markdown(f"Model parameters: {n_parameters:.3f}M")

    start_token_id: int = dataset.encoder.token_to_id["<CLS>"]
    pad_token_id: int = dataset.encoder.token_to_id["<PAD>"]
    config = T5Config(
        vocab_size=vocab_size(train_cfg),
        decoder_start_token_id=start_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=pad_token_id,
        use_cache=False,
        d_model=train_cfg.model.d_model,
        d_kv=train_cfg.model.d_kv,
        d_ff=train_cfg.model.d_ff,
        num_layers=train_cfg.model.num_layers,
        num_heads=train_cfg.model.num_heads,
    )

    model = T5ForConditionalGeneration(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(DEVICE)

    n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    st.markdown(f"Model parameters: {n_parameters:.3f}M")

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
        generated_velocity = tokenizer.decode_tgt(generated_velocity)

        # TODO start here
        # Reconstruct the sequence as recorded
        midi_columns = ["pitch", "start", "end", "duration", "velocity"]
        true_notes = pd.DataFrame({c: record[c] for c in midi_columns})
        true_piece = MidiPiece(df=true_notes, source=record_source)
        true_piece.time_shift(-true_piece.df.start.min())

        pred_piece_df = true_piece.df.copy()

        # change untokenized velocities to model predictions
        pred_piece_df["velocity"] = generated_velocity
        pred_piece_df["velocity"] = pred_piece_df["velocity"].fillna(0)

        # create quantized piece with predicted velocities
        pred_piece = MidiPiece(pred_piece_df)

        pred_piece.source = true_piece.source.copy()

        # create a dashboard
        st.json(record_source, expanded=False)
        cols = st.columns(2)
        with cols[0]:
            # Unchanged
            fig = ff.view.draw_pianoroll_with_velocities(true_piece)
            st.pyplot(fig)
            from_fortepyan(true_piece)

        with cols[1]:
            # Predicted
            fig = ff.view.draw_pianoroll_with_velocities(pred_piece)
            st.pyplot(fig)
            from_fortepyan(pred_piece)


if __name__ == "__main__":
    main()
