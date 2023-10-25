import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from pipelines.BART.main import main as bart_training

import wandb
from data.dataset import load_cache_dataset
from pipelines.T5.main import main as t5_training


def initialize_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


@hydra.main(version_base=None, config_path="configs", config_name="T5velocity")
def main(cfg: DictConfig):
    if cfg.log:
        initialize_wandb(cfg)

    if cfg.overfit:
        split_slice = "[:1]"
    else:
        split_slice = ""
    np.random.seed(cfg.seed)
    train_translation_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name=cfg.dataset_name,
        split="train" + split_slice,
    )
    val_translation_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name="roszcz/maestro-v1-sustain",
        split=f"validation{split_slice}+test{split_slice}",
    )
    if cfg.model_name == "T5":
        t5_training(cfg, train_translation_dataset, val_translation_dataset)


if __name__ == "__main__":
    main()
