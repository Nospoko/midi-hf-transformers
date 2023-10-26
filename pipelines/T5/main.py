from datasets import Dataset
from omegaconf import DictConfig
from transformers import T5Config, T5ForConditionalGeneration

from utils import vocab_size
from training_utils import train_model
from data.dataset import MaskedMidiDataset, MyTokenizedMidiDataset
from data.midiencoder import VelocityEncoder, QuantizedMidiEncoder
from data.multitokencoder import MultiStartEncoder, MultiVelocityEncoder


def main(
    cfg: DictConfig,
    train_translation_dataset: Dataset,
    val_translation_dataset: Dataset,
):
    config = T5Config(
        vocab_size=vocab_size(cfg),
        decoder_start_token_id=0,
        use_cache=False,
        d_model=cfg.model.d_model,
        d_kv=cfg.model.d_kv,
        d_ff=cfg.model.d_ff,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
    )

    model = T5ForConditionalGeneration(config)

    if cfg.target == "denoise":
        train_dataset, val_dataset = create_masked_datasets(
            cfg=cfg,
            train_translation_dataset=train_translation_dataset,
            val_translation_dataset=val_translation_dataset,
        )
    else:
        train_dataset, val_dataset = create_datasets(
            cfg=cfg,
            train_translation_dataset=train_translation_dataset,
            val_translation_dataset=val_translation_dataset,
        )

    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
    )

    print(cfg.run_name)


def create_datasets(
    cfg: DictConfig,
    train_translation_dataset: Dataset,
    val_translation_dataset: Dataset,
) -> tuple[MyTokenizedMidiDataset, MyTokenizedMidiDataset]:
    if cfg.tokens_per_note == "multiple":
        if cfg.target == "velocity":
            tokenizer = MultiVelocityEncoder(
                quantization_cfg=cfg.dataset.quantization,
                time_quantization_method=cfg.time_quantization_method,
            )
        else:
            tokenizer = MultiStartEncoder(
                quantization_cfg=cfg.dataset.quantization,
                time_quantization_method=cfg.time_quantization_method,
                tgt_bins=cfg.start_bins,
            )
    else:
        tokenizer = VelocityEncoder(
            quantization_cfg=cfg.dataset.quantization,
            time_quantization_method=cfg.time_quantization_method,
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
    return train_dataset, val_dataset


def create_masked_datasets(
    cfg: DictConfig,
    train_translation_dataset: Dataset,
    val_translation_dataset: Dataset,
) -> tuple[MyTokenizedMidiDataset, MyTokenizedMidiDataset]:
    if cfg.tokens_per_note == "multiple":
        base_encoder = MultiStartEncoder(cfg.dataset.quantization, cfg.time_quantization_method, tgt_bins=0)
    else:
        base_encoder = QuantizedMidiEncoder(cfg.dataset.quantization, cfg.time_quantization_method)

    train_dataset = MaskedMidiDataset(
        dataset=train_translation_dataset,
        dataset_cfg=cfg.dataset,
        base_encoder=base_encoder,
        masking_probability=cfg.masking_probability,
    )

    val_dataset = MaskedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=cfg.dataset,
        base_encoder=base_encoder,
        masking_probability=cfg.masking_probability,
    )

    return train_dataset, val_dataset
