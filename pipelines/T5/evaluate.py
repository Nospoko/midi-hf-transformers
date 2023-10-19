import glob

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from transformers import T5Config, T5ForConditionalGeneration

from utils import vocab_size
from data.tokenizer import MultiStartEncoder, MultiVelocityEncoder
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset


def load_model_checkpoint(cfg: DictConfig) -> dict:
    model_path = None
    for file in glob.glob("checkpoints/*/*.pt"):
        if cfg.run_name in file:
            model_path = file
    if model_path is None:
        raise FileNotFoundError()
    return torch.load(model_path, map_location=cfg.device)


@hydra.main(version_base=None, config_path="../../configs", config_name="T5eval")
@torch.no_grad()
def main(cfg: DictConfig):
    checkpoint = load_model_checkpoint(cfg)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    val_translation_dataset = load_cache_dataset(
        dataset_cfg=train_cfg.dataset,
        dataset_name="roszcz/maestro-v1-sustain",
        split=cfg.split,
    )

    config = T5Config(
        vocab_size=vocab_size(train_cfg),
        decoder_start_token_id=0,
        use_cache=False,
    )

    model = T5ForConditionalGeneration(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(cfg.device)

    keys = ["pitch"] + [f"{key}_bin" for key in train_cfg.dataset.quantization]
    if train_cfg.target == "velocity":
        tokenizer = MultiVelocityEncoder(train_cfg.dataset.quantization, keys=keys)
    else:
        tokenizer = MultiStartEncoder(
            quantization_cfg=train_cfg.dataset.quantization,
            keys=keys,
            tgt_bins=train_cfg.start_bins,
        )

    val_dataset = MyTokenizedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=train_cfg.dataset,
        encoder=tokenizer,
    )

    record = val_dataset[cfg.idx]
    input_ids = record["source_token_ids"].unsqueeze(0)
    out = model.generate(input_ids, max_length=train_cfg.max_padding)
    print(input_ids, "\n", out, "\n", record["target_token_ids"])


if __name__ == "__main__":
    main()
