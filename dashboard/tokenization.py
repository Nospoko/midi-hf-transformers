import os

import numpy as np
import streamlit as st
from fortepyan import MidiPiece
from datasets import load_dataset

from utils import piece_av_files
from data.quantizer import MidiQuantizer, MidiCTQuantizer
from data.tokenizer import MultiVelocityEncoder


def CT_tokenization_review_dashboard():
    seq_one = """
    ## CT Quantization
    New quantization method allows you to quantize start time without calculating dstart value first.
    `sequence_duration` is maximum distance between earliest and latest start in a musical sequence. This length
    is then divided linearly into n_start_bins sections.
    Edges of these sections are our bin edges for start quantization.
    """

    st.markdown(seq_one)
    st.markdown("### Quantization settings")

    n_start_bins = st.number_input(label="n_start_bins", value=400)
    n_duration_bins = st.number_input(label="n_duration_bins", value=3)
    n_velocity_bins = st.number_input(label="n_velocity_bins", value=3)
    sequence_duration = st.number_input(label="sequence_duration", value=5)

    quantizer = MidiCTQuantizer(
        n_start_bins=n_start_bins,
        n_duration_bins=n_duration_bins,
        n_velocity_bins=n_velocity_bins,
        sequence_duration=sequence_duration,
    )

    split = "test"
    dataset_name = "roszcz/maestro-v1-sustain"
    dataset = load_dataset(dataset_name, split=split)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Quantized")

    seed = st.number_input(label="random seed", value=137)
    np.random.seed(seed=seed)
    n_samples = 2
    ids = np.random.randint(len(dataset), size=n_samples)
    for idx in ids:
        piece = MidiPiece.from_huggingface(dataset[int(idx)])

        start = np.random.randint(len(piece.df["pitch"]))
        finish = start
        while piece.df["start"][finish] - piece.df["start"][start] < sequence_duration:
            finish += 1
        st.markdown(f"There are {finish - start} notes in this sequence")

        piece = piece[start:finish]
        quantized_piece = quantizer.quantize_piece(piece)

        av_dir = "tmp/dashboard/common"
        bins = f"{n_start_bins}-{n_duration_bins}-{n_velocity_bins}"
        save_name = f"ct-{dataset_name}-{split}-{idx}".replace("/", "_")

        save_base_gt = os.path.join(av_dir, f"{save_name}")
        gt_paths = piece_av_files(piece, save_base=save_base_gt)

        save_base_quantized = os.path.join(av_dir, f"{save_name}-{bins}")
        quantized_piece_paths = piece_av_files(quantized_piece, save_base=save_base_quantized)

        st.json(piece.source)
        cols = st.columns(2)
        with cols[0]:
            st.image(gt_paths["pianoroll_path"])
            st.audio(gt_paths["mp3_path"])

        with cols[1]:
            st.image(quantized_piece_paths["pianoroll_path"])
            st.audio(quantized_piece_paths["mp3_path"])

    seq_two = """
    ## MultiTokEncoder
    T5 models have equal input and output layer sizes. This means they are using the same vocabulary for source and
    target encoding. 
    
    MultiTokEncoders have methods for tokenizing records to both source and target sequences.
    `tokenize_src` will output sequence of tokens, e.g. for a dataframe of quantized notes:
    |pitch|start_bin|duration_bin|velocity_bin|velocity|
    |-----|---------|------------|------------|--------|
    |78|155|3|7|95|
    |66|157|6|5|62|
    |68|158|5|2|20|
    
    It will output a sequence of tokens:
    ``` 
    ["<PITCH-78>", "<TIME-155-3>", "<VEL-7>", "<PITCH-66>", \
    "<TIME-157-6>", "<VEL-5>", "<PITCH-68>", "<TIME-158-5>", "<VEL-2>"]
    ```
    So three tokens for each note. When untokenizing the tokenizer asserts tokens are in correct order 
    (pitch, time, velocity)
    
    `MultiVelocityEncoder`'s `tokenize_tgt` will output from the same dataframe:
    ```
    ["<VEL-95>", "<VEL-62>", "<VEL-20>"]
    ```
    `MultiStartEncoder`'s `tokenize_tgt` will work similarly, but it will quantize `start` into `start_bins` first. 
    """

    st.markdown(seq_two)


def classic_tokenization_review_dashboard():
    st.markdown("### Quantization settings")

    n_dstart_bins = st.number_input(label="n_dstart_bins", value=3)
    n_duration_bins = st.number_input(label="n_duration_bins", value=3)
    n_velocity_bins = st.number_input(label="n_velocity_bins", value=3)

    quantizer = MidiQuantizer(
        n_dstart_bins=n_dstart_bins,
        n_duration_bins=n_duration_bins,
        n_velocity_bins=n_velocity_bins,
    )

    split = "train"
    dataset_name = "roszcz/maestro-v1-sustain"
    dataset = load_dataset(dataset_name, split=split)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Quantized")

    np.random.seed(137)
    n_samples = 5
    ids = np.random.randint(len(dataset), size=n_samples)
    for idx in ids:
        piece = MidiPiece.from_huggingface(dataset[int(idx)])
        quantized_piece = quantizer.quantize_piece(piece)

        av_dir = "tmp/dashboard/common"
        bins = f"{n_dstart_bins}-{n_duration_bins}-{n_velocity_bins}"
        save_name = f"{dataset_name}-{split}-{idx}".replace("/", "_")

        save_base_gt = os.path.join(av_dir, f"{save_name}")
        gt_paths = piece_av_files(piece, save_base=save_base_gt)

        save_base_quantized = os.path.join(av_dir, f"{save_name}-{bins}")
        quantized_piece_paths = piece_av_files(quantized_piece, save_base=save_base_quantized)

        st.json(piece.source)
        cols = st.columns(2)
        with cols[0]:
            st.image(gt_paths["pianoroll_path"])
            st.audio(gt_paths["mp3_path"])

        with cols[1]:
            st.image(quantized_piece_paths["pianoroll_path"])
            st.audio(quantized_piece_paths["mp3_path"])


def tokenization_review_dashboard():
    display_mode = st.selectbox(label="quantization method", options=["continuous start time", "classic"])
    if display_mode == "classic":
        classic_tokenization_review_dashboard()
    else:
        CT_tokenization_review_dashboard()


if __name__ == "__main__":
    tokenization_review_dashboard()
