import hydra
import wandb
from omegaconf import OmegaConf, DictConfig
from transformers import T5Config, T5ForConditionalGeneration

from utils import vocab_size
from training_utils import train_model
from data.tokenizer import MultiStartEncoder, MultiVelocityEncoder
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset


def initialize_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


@hydra.main(version_base=None, config_path="configs", config_name="T5velocity")
def main(cfg: DictConfig):
    keys = ["pitch"] + [f"{key}_bin" for key in cfg.dataset.quantization]
    if cfg.target == "velocity":
        tokenizer = MultiVelocityEncoder(cfg.dataset.quantization, keys=keys)
    else:
        tokenizer = MultiStartEncoder(quantization_cfg=cfg.dataset.quantization, keys=keys, tgt_bins=cfg.start_bins)

    config = T5Config(
        vocab_size=vocab_size(cfg),
        decoder_start_token_id=0,
        use_cache=False,
    )
    model = T5ForConditionalGeneration(config)

    train_translation_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name=cfg.dataset_name,
        split="train",
    )
    val_translation_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name="roszcz/maestro-v1-sustain",
        split="validation+test",
    )

    train_dataset = MyTokenizedMidiDataset(
        dataset=train_translation_dataset,
        dataset_cfg=cfg.dataset,
        encoder=tokenizer,
    )
    val_dataset = MyTokenizedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=cfg.dataset,
        encoder=tokenizer,
    )
    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
