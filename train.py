import hydra
import numpy as np
from datasets import concatenate_datasets
from omegaconf import OmegaConf, DictConfig

import wandb
from data.dataset import load_cache_dataset
from pipelines.T5.main import main as t5_training
from pipelines.BART.main import main as bart_training


def initialize_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def load_train_dataset(cfg: DictConfig):
    datasets = []
    for name in cfg.dataset_name.split("+"):
        dataset = load_cache_dataset(
            dataset_cfg=cfg.dataset,
            dataset_name=name,
            split="train",
        )

        datasets.append(dataset)
    train_dataset = concatenate_datasets(datasets)

    return train_dataset


@hydra.main(version_base=None, config_path="configs", config_name="T5velocity")
def main(cfg: DictConfig):
    if cfg.log:
        initialize_wandb(cfg)

    np.random.seed(cfg.seed)
    while True:
        try:
            train_translation_dataset = load_train_dataset(cfg)

            val_translation_dataset = load_cache_dataset(
                dataset_cfg=cfg.dataset,
                dataset_name="roszcz/maestro-v1-sustain",
                split="validation+test",
            )
            break
        except ConnectionError:
            print("Connection error, trying again...")
    if cfg.model_name == "T5":
        t5_training(cfg, train_translation_dataset, val_translation_dataset)
    elif cfg.model_name == "BART":
        bart_training(cfg, train_translation_dataset, val_translation_dataset)


if __name__ == "__main__":
    main()
