import hydra
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from transformers import T5Config, T5ForConditionalGeneration

from data.tokenizer import MultiStartEncoder, MultiVelocityEncoder
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset


def collate(batch, max_padding: int = 128, pad_idx: int = 1):
    """
    Add padding to a batch of records.
    """
    src_list, tgt_list = [], []
    for record in batch:
        src = record["source_token_ids"]
        tgt = record["target_token_ids"]
        src_list.append(
            pad(
                src,
                (
                    0,
                    max_padding - len(src),
                ),
                value=pad_idx,
            )
        )
        tgt_list.append(
            pad(
                tgt,
                (0, max_padding - len(tgt)),
                value=pad_idx,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return {"source_token_ids": src, "target_token_ids": tgt}


@hydra.main(version_base=None, config_path="configs", config_name="T5velocity")
def main(cfg: DictConfig):
    keys = ["pitch"] + [f"{key}_bin" for key in cfg.dataset.quantization]
    if cfg.target == "velocity":
        tokenizer = MultiVelocityEncoder(cfg.dataset.quantization, keys=keys)
    else:
        tokenizer = MultiStartEncoder(quantization_cfg=cfg.dataset.quantization, keys=keys, tgt_bins=cfg.start_bins)
    pad_idx = tokenizer.token_to_id["<pad>"]

    def collate_fn(batch):
        return collate(batch, cfg.max_padding, pad_idx)

    config = T5Config(
        vocab_size=tokenizer.vocab_size,
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

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.98), eps=1e-9)
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

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    pbar = tqdm(train_dataloader)
    for batch in pbar:
        attention_mask = batch["source_token_ids"] != pad_idx
        # -100 will be ignored by the loss
        batch["target_token_ids"][batch["target_token_ids"] == pad_idx] = -100
        loss = model(
            input_ids=batch["source_token_ids"],
            attention_mask=attention_mask,
            labels=batch["target_token_ids"],
        ).loss
        loss.backward()
        optimizer.step()
        pbar.set_description(f"{loss.item():0.3f}")

    pbar = tqdm(val_dataloader)
    with torch.no_grad():
        val_loss = 0
        for batch in pbar:
            attention_mask = batch["source_token_ids"] != pad_idx
            loss = model(
                input_ids=batch["source_token_ids"],
                attention_mask=attention_mask,
                labels=batch["target_token_ids"],
            ).loss
            val_loss += loss.item
            pbar.set_description(f"{loss.item():0.3f}")
        val_loss /= len(val_dataloader)
    print(val_loss)
    save_checkpoint(model=model, optimizer=optimizer, cfg=cfg)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
):
    path = f"checkpoints/{cfg.target}/{cfg.run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": OmegaConf.to_object(cfg),
        },
        path,
    )
    print("Saved!", cfg.run_name)


if __name__ == "__main__":
    main()
