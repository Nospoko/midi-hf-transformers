import torch
from datasets import Dataset
from omegaconf import OmegaConf, DictConfig
from transformers import T5Config, T5ForConditionalGeneration

from utils import vocab_size
from training_utils import train_model
from data.dataset import MaskedMidiDataset, MyTokenizedMidiDataset
from data.midiencoder import VelocityEncoder, QuantizedMidiEncoder
from data.maskedmidiencoder import MaskedMidiEncoder, MaskedNoteEncoder
from data.multitokencoder import MultiMidiEncoder, MultiStartEncoder, MultiVelocityEncoder


def main(
    cfg: DictConfig,
    train_translation_dataset: Dataset,
    val_translation_dataset: Dataset,
):
    checkpoint = None
    if cfg.target == "denoise":
        train_dataset, val_dataset = create_masked_datasets(
            cfg=cfg,
            train_translation_dataset=train_translation_dataset,
            val_translation_dataset=val_translation_dataset,
        )
    elif cfg.train.finetune:
        checkpoint = torch.load(f"checkpoints/denoise/{cfg.pretrained_checkpoint}", map_location=cfg.device)
        pretrain_cfg = OmegaConf.create(checkpoint["cfg"])
        # make current cfg fit pre-train cfg
        pretrain_cfg.device = cfg.device
        pretrain_cfg.target = cfg.target
        pretrain_cfg.train.finetune = True
        pretrain_cfg.run_name = cfg.run_name
        cfg = pretrain_cfg

        train_dataset, val_dataset = create_datasets_finetune(
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

    start_token_id: int = train_dataset.encoder.token_to_id["<CLS>"]
    pad_token_id: int = train_dataset.encoder.token_to_id["<PAD>"]
    config = T5Config(
        vocab_size=vocab_size(cfg),
        decoder_start_token_id=start_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=pad_token_id,
        use_cache=False,
        d_model=cfg.model.d_model,
        d_kv=cfg.model.d_kv,
        d_ff=cfg.model.d_ff,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
    )

    model = T5ForConditionalGeneration(config)
    if checkpoint is not None:
        # Pre-trained model has to be trained with the same vocab_size as our model.
        # To do that, pre-trained model has to be trained using a base_encoder,
        # initialized the same way as our tokenizer.
        model.load_state_dict(checkpoint["model_state_dict"])

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
        base_encoder = MultiMidiEncoder(
            quantization_cfg=cfg.dataset.quantization, time_quantization_method=cfg.time_quantization_method
        )
    else:
        base_encoder = QuantizedMidiEncoder(cfg.dataset.quantization, cfg.time_quantization_method)
    if cfg.mask == "notes":
        encoder = MaskedNoteEncoder(base_encoder=base_encoder, masking_probability=cfg.masking_probability)
    else:
        encoder = MaskedMidiEncoder(base_encoder=base_encoder, masking_probability=cfg.masking_probability)

    train_dataset = MaskedMidiDataset(
        dataset=train_translation_dataset,
        dataset_cfg=cfg.dataset,
        base_encoder=base_encoder,
        encoder=encoder,
    )

    val_dataset = MaskedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=cfg.dataset,
        base_encoder=base_encoder,
        encoder=encoder,
    )

    return train_dataset, val_dataset


def create_datasets_finetune(
    cfg: DictConfig,
    train_translation_dataset: Dataset,
    val_translation_dataset: Dataset,
) -> tuple[MyTokenizedMidiDataset, MyTokenizedMidiDataset]:
    tokenizer = MultiMidiEncoder(
        quantization_cfg=cfg.dataset.quantization,
        time_quantization_method=cfg.time_quantization_method,
    )
    pretraining_tokenizer = MaskedMidiEncoder(
        base_encoder=tokenizer,
    )
    # use the same token ids as used during pre-training
    tokenizer.vocab = pretraining_tokenizer.vocab
    tokenizer.token_to_id = pretraining_tokenizer.token_to_id
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
