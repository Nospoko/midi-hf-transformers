from datasets import Dataset
from omegaconf import DictConfig
from transformers import BartConfig, BartForConditionalGeneration

from utils import vocab_size
from training_utils import train_model
from data.dataset import MyTokenizedMidiDataset
from data.multitokencoder import MaskedMidiEncoder


def main(
    cfg: DictConfig,
    train_translation_dataset: Dataset,
    val_translation_dataset: Dataset,
):
    config = BartConfig(
        vocab_size=vocab_size(cfg),
        decoder_start_token_id=0,
        use_cache=False,
    )

    model = BartForConditionalGeneration(config)
    print(model.num_parameters())

    tokenizer = MaskedMidiEncoder(
        quantization_cfg=cfg.dataset.quantization,
        time_quantization_method=cfg.time_quantization_method,
        masking_probability=cfg.masking_probability,
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

    print(cfg.run_name)
