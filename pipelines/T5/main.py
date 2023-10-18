from datasets import Dataset
from omegaconf import DictConfig
from transformers import T5Config, T5ForConditionalGeneration

from utils import vocab_size
from training_utils import train_model
from data.dataset import MyTokenizedMidiDataset
from data.tokenizer import MultiStartEncoder, MultiVelocityEncoder


def main(
    cfg: DictConfig,
    train_translation_dataset: Dataset,
    val_translation_dataset: Dataset,
):
    config = T5Config(
        vocab_size=vocab_size(cfg),
        decoder_start_token_id=0,
        use_cache=False,
    )

    model = T5ForConditionalGeneration(config)

    keys = ["pitch"] + [f"{key}_bin" for key in cfg.dataset.quantization]
    if cfg.target == "velocity":
        tokenizer = MultiVelocityEncoder(cfg.dataset.quantization, keys=keys)
    else:
        tokenizer = MultiStartEncoder(quantization_cfg=cfg.dataset.quantization, keys=keys, tgt_bins=cfg.start_bins)

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

    print(cfg.run_name)
