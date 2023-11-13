from huggingface_hub import hf_hub_download

FILENAME_VELOCITY = "velocity-T5-2023-11-11-10-29.pt"
FILENAME_DENOISE = "midi-T5-2023-11-11-10-29.pt"

hf_hub_download(
    repo_id="wmatejuk/midi-T5-velocity",
    filename=FILENAME_VELOCITY,
    local_dir="checkpoints/velocity",
    local_dir_use_symlinks=False,
)

hf_hub_download(
    repo_id="wmatejuk/midi-T5-denoise",
    filename=FILENAME_DENOISE,
    local_dir="checkpoints/denoise",
    local_dir_use_symlinks=False,
)
