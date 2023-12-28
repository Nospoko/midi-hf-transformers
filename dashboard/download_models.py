from huggingface_hub import hf_hub_download

# FILENAME_VELOCITY = "velocity-T5-2023-11-11-10-29.pt"
FILENAME_DENOISE = "midi-bart-2023-12-26-19-05.pt"

hf_hub_download(
    repo_id="wmatejuk/midi-bart-denoise",
    filename=FILENAME_DENOISE,
    local_dir="checkpoints/denoise",
    local_dir_use_symlinks=False,
)
