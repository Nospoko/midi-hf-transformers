import os
import glob
import json

import torch
import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
from datasets import Dataset
from fortepyan import MidiPiece
from omegaconf import OmegaConf, DictConfig
from streamlit_pianoroll import from_fortepyan
from transformers import T5Config, BartConfig, T5ForConditionalGeneration, BartForConditionalGeneration

from utils import vocab_size
from data.midiencoder import QuantizedMidiEncoder
from data.multitokencoder import MultiMidiEncoder
from data.quantizer import MidiQuantizer, MidiATQuantizer
from data.dataset import MaskedMidiDataset, load_cache_dataset
from data.maskedmidiencoder import MaskedMidiEncoder, MaskedNoteEncoder

# Set the layout of the Streamlit page
st.set_page_config(layout="wide", page_title="Denoise MIDI", page_icon=":musical_keyboard")

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
        options = glob.glob("checkpoints/denoise/*.pt")
        options.sort(reverse=True)
        checkpoint_path = st.selectbox(label="model", options=options)
        st.markdown("Selected checkpoint:")
        st.markdown(checkpoint_path)

    # Load:
    checkpoint: dict = torch.load(checkpoint_path, map_location=DEVICE)

    # - original config
    train_cfg: DictConfig = OmegaConf.create(checkpoint["cfg"])
    train_cfg.device = DEVICE

    # - - for model
    st.markdown("Model config:")
    model_params: dict = OmegaConf.to_container(train_cfg.model)
    st.json(model_params, expanded=False)

    # - - for dataset
    dataset_params: dict = OmegaConf.to_container(train_cfg.dataset)
    st.markdown("Dataset config:")
    st.json(dataset_params, expanded=True)

    # Folder to render audio and video
    model_dir: str = f"tmp/dashboard/{train_cfg.run_name}"

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
    dataset_cfg: DictConfig = train_cfg.dataset
    dataset_name: str = st.text_input(label="dataset", value=train_cfg.dataset_name)
    split: str = st.text_input(label="split", value="test")

    random_seed: int = st.selectbox(label="random seed", options=range(20))

    # load translation dataset and create MyTokenizedMidiDataset
    val_translation_dataset: Dataset = load_cache_dataset(
        dataset_cfg=dataset_cfg,
        dataset_name=dataset_name,
        split=split,
    )
    if train_cfg.time_quantization_method == "start":
        quantizer = MidiATQuantizer(
            n_duration_bins=dataset_cfg.quantization.duration,
            n_velocity_bins=dataset_cfg.quantization.velocity,
            n_start_bins=dataset_cfg.quantization.start,
            sequence_duration=dataset_cfg.sequence_duration,
        )
    else:
        quantizer = MidiQuantizer(
            n_velocity_bins=dataset_cfg.quantization.velocity,
            n_duration_bins=dataset_cfg.quantization.duration,
            n_dstart_bins=dataset_cfg.quantization.dstart,
        )
    if train_cfg.tokens_per_note == "multiple":
        base_tokenizer = MultiMidiEncoder(
            quantization_cfg=train_cfg.dataset.quantization,
            time_quantization_method=train_cfg.time_quantization_method,
        )
    else:
        base_tokenizer = QuantizedMidiEncoder(
            quantization_cfg=train_cfg.dataset.quantization,
            time_quantization_method=train_cfg.time_quantization_method,
        )

    if "mask" in train_cfg:
        if train_cfg.mask == "notes":
            encoder = MaskedNoteEncoder(base_encoder=base_tokenizer, masking_probability=train_cfg.masking_probability)
        else:
            encoder = MaskedMidiEncoder(base_encoder=base_tokenizer, masking_probability=train_cfg.masking_probability)
    else:
        encoder = MaskedMidiEncoder(base_encoder=base_tokenizer, masking_probability=train_cfg.masking_probability)

    dataset = MaskedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=train_cfg.dataset,
        base_encoder=base_tokenizer,
        encoder=encoder,
    )

    start_token_id: int = dataset.encoder.token_to_id["<CLS>"]
    pad_token_id: int = dataset.encoder.token_to_id["<PAD>"]
    if train_cfg.model_name == "T5":
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
    else:
        config = BartConfig(
            vocab_size=vocab_size(train_cfg),
            decoder_start_token_id=start_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
            use_cache=False,
            d_model=train_cfg.model.d_model,
            encoder_layers=train_cfg.model.encoder_layers,
            decoder_layers=train_cfg.model.decoder_layers,
            encoder_ffn_dim=train_cfg.model.encoder_ffn_dim,
            decoder_ffn_dim=train_cfg.model.decoder_ffn_dim,
            encoder_attention_heads=train_cfg.model.encoder_attention_heads,
            decoder_attention_heads=train_cfg.model.decoder_attention_heads,
        )

        model = BartForConditionalGeneration(config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(DEVICE)

    n_parameters: float = sum(p.numel() for p in model.parameters()) / 1e6
    st.markdown(f"Model parameters: {n_parameters:.3f}M")

    n_samples: int = 5
    np.random.seed(random_seed)
    ids: int | np.ndarray[int] = np.random.randint(len(dataset), size=n_samples)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Predicted")

    # predict velocities and get src, tgt and model output
    print("Making predictions ...")

    # widget id for streamlit_pianoroll widget
    key = 0
    for record_id in ids:
        # Numpy to int :(
        record: dict = dataset.get_complete_record(int(record_id))
        record_source: dict = json.loads(record["source"])
        src_token_ids: torch.Tensor = record["source_token_ids"]

        generated_token_ids: torch.Tensor = model.generate(src_token_ids.unsqueeze(0), max_length=128)
        generated_token_ids = generated_token_ids.squeeze(0)

        # Reconstruct the sequence as recorded
        midi_columns = ["pitch", "start", "end", "duration", "velocity"]
        true_notes = pd.DataFrame({c: record[c] for c in midi_columns})
        true_piece = MidiPiece(df=true_notes, source=record_source)
        true_piece.time_shift(-true_piece.df.start.min())
        try:
            generated_df: pd.DataFrame = encoder.decode(src_token_ids, generated_token_ids)
            df = quantizer.apply_quantization(generated_df)
            df["mask"] = generated_df["mask"]
            # create quantized piece with predicted notes
            pred_piece = MidiPiece(df)
            unmasked_notes_df = generated_df[generated_df["mask"]]
            unmasked_notes_piece = MidiPiece(unmasked_notes_df)

        except ValueError:
            # create an empty piece if the model did not generate the structure correctly
            generated_df = pd.DataFrame([[23.0, 1.0, 1.0, 1.0, 1.0]], columns=midi_columns)
            generated_df["mask"] = [False]
            pred_piece = MidiPiece(generated_df)

        pred_piece.source = true_piece.source.copy()

        # create a dashboard
        st.json(record_source, expanded=False)
        cols = st.columns(2)

        source_tokens: list[str] = [dataset.encoder.vocab[idx] for idx in src_token_ids]
        tgt_tokens: list[str] = [dataset.encoder.vocab[idx] for idx in record["target_token_ids"]]
        generated_tokens: list[str] = [dataset.encoder.vocab[idx] for idx in generated_token_ids]

        with cols[0]:
            fig = ff.view.draw_pianoroll_with_velocities(true_piece)
            st.pyplot(fig)
            from_fortepyan(true_piece, key=key)
            # Unchanged
            st.markdown("**Source tokens:**")
            st.markdown(source_tokens)
            st.markdown("**Target tokens:**")
            st.markdown(tgt_tokens)

        with cols[1]:
            # Predicted
            fig = ff.view.draw_dual_pianoroll(pred_piece)
            st.pyplot(fig)
            from_fortepyan(pred_piece, secondary_piece=unmasked_notes_piece, key=key + 1)
            st.markdown("**Predicted tokens:**")
            st.markdown(generated_tokens)
        key += 2


if __name__ == "__main__":
    main()
