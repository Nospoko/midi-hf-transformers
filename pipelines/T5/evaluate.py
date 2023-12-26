import glob

import hydra
import torch
from tqdm import tqdm
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

    if train_cfg.target == "velocity":
        tokenizer = MultiVelocityEncoder(
            quantization_cfg=train_cfg.dataset.quantization,
            time_quantization_method=train_cfg.time_quantization_method,
        )
    else:
        tokenizer = MultiStartEncoder(
            quantization_cfg=train_cfg.dataset.quantization,
            time_quantization_method=train_cfg.time_quantization_method,
            tgt_bins=train_cfg.start_bins,
        )

    val_dataset = MyTokenizedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=train_cfg.dataset,
        encoder=tokenizer,
    )

    total_dist = 0
    pbar = tqdm(val_dataset)
    it = 0

    for record in pbar:
        it += 1

        input_ids = record["source_token_ids"].unsqueeze(0)
        sequence_len = len(record["source_token_ids"]) // 3 + 1
        out = model.generate(input_ids, max_length=sequence_len)

        out = out[: len(record["target_token_ids"])]
        dist = torch.dist(out.to(float), record["target_token_ids"].to(float)).data
        total_dist += dist
        pbar.set_description(f"avg_dist: {total_dist / it:0.3f}")
    print(total_dist / len(val_dataset))


if __name__ == "__main__":
    main()
