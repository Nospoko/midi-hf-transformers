import time
from typing import Iterable

import torch
import einops
import torch.nn as nn
from tqdm import tqdm
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from torch.optim.lr_scheduler import LambdaLR

import wandb
from data.dataset import MyTokenizedMidiDataset
from utils import learning_rate_schedule, calculate_average_distance


def train_model(
    model: nn.Module,
    train_dataset: MyTokenizedMidiDataset,
    val_dataset: MyTokenizedMidiDataset,
    cfg: DictConfig,
) -> nn.Module:
    model.to(cfg.device)
    pad_idx = train_dataset.encoder.token_to_id["<PAD>"]

    def collate_fn(batch):
        return collate(batch, pad_idx)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.base_lr)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: learning_rate_schedule(step, warmup=cfg.train.warmup),
    )
    best_test_loss = float("inf")
    for epoch in range(cfg.train.num_epochs):
        model.train()
        print(f"Epoch {epoch}", flush=True)

        # Train model for one epoch
        t_loss, t_dist = train_epoch(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            pad_idx=pad_idx,
            accum_iter=cfg.train.accum_iter,
            log=cfg.log,
            log_frequency=cfg.log_frequency,
            device=cfg.device,
        )

        print(f"Epoch {epoch} Validation", flush=True)
        model.eval()
        # Evaluate the model on validation set
        v_loss, v_dist = val_epoch(
            dataloader=val_dataloader,
            model=model,
            device=cfg.device,
            pad_idx=pad_idx,
        )

        if v_loss <= best_test_loss:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                cfg=cfg,
            )
            best_test_loss = v_loss

        # Log validation and training losses
        if cfg.log:
            wandb.log(
                {
                    "val/loss_epoch": v_loss,
                    "val/dist_epoch": v_dist,
                    "train/loss_epoch": t_loss,
                    "train/dist_epoch": t_dist,
                    "epoch": epoch,
                }
            )

    return model


def collate(batch, pad_idx: int = 1):
    """
    Add padding to a batch of records.
    """
    src_list, tgt_list = [], []
    max_padding = max([len(record["source_token_ids"]) for record in batch])
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


def train_epoch(
    dataloader: Iterable,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    pad_idx: int = 1,
    accum_iter: int = 1,
    log_frequency: int = 10,
    log: bool = False,
    device: str = "cpu",
) -> tuple[float, float]:
    start = time.time()
    total_loss = 0
    total_dist = 0
    tokens = 0
    n_accum = 0
    it = 0

    # create progress bar
    steps = len(dataloader)
    progress_bar = tqdm(dataloader, total=steps)
    for batch in progress_bar:
        src = batch["source_token_ids"].to(device)
        tgt = batch["target_token_ids"].to(device)
        n_tokens = tgt.numel()
        tgt[tgt == pad_idx] = -100
        tgt[tgt == 0] = -100
        attention_mask = src != pad_idx

        outputs = model(
            input_ids=src,
            attention_mask=attention_mask,
            labels=tgt,
        )
        out = outputs.logits

        out_rearranged = einops.rearrange(out, "b n d -> (b n) d")
        target = einops.rearrange(tgt, "b n -> (b n)")
        loss = outputs.loss
        loss.backward()

        dist = calculate_average_distance(out_rearranged, target, pad_idx=pad_idx)

        # Update the model parameters and optimizer gradients every `accum_iter` iterations
        if it % accum_iter == 0 or it == steps - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
        it += 1

        # Update learning learning_rate_schedule lr_scheduler
        lr_scheduler.step()

        # Update loss and token counts
        loss_item = loss.item()
        total_loss += loss.item()
        tokens += n_tokens
        total_dist += dist

        # log metrics every log_frequency steps
        if it % log_frequency == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            tok_rate = tokens / elapsed
            progress_bar.set_description(
                f"Step: {it:6d}/{steps} | acc_step: {n_accum:3d} | loss: {loss_item:6.2f} | dist: {dist:6.2f}"
                + f"| tps: {tok_rate:7.1f} | lr: {lr:6.1e}"
            )

            # log the loss each to Weights and Biases
            if log:
                wandb.log({"train/loss_step": loss.item()})

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader), total_dist / len(dataloader)


@torch.no_grad()
def val_epoch(
    dataloader: Iterable,
    model: nn.Module,
    pad_idx: int = 1,
    device: str = "cpu",
) -> tuple[float, float]:
    total_tokens = 0
    total_loss = 0
    tokens = 0
    total_dist = 0

    for batch in tqdm(dataloader):
        src = batch["source_token_ids"].to(device)
        tgt = batch["target_token_ids"].to(device)
        n_tokens = tgt.numel()
        tgt[tgt == pad_idx] = -100
        attention_mask = src != pad_idx

        outputs = model(
            input_ids=src,
            attention_mask=attention_mask,
            labels=tgt,
        )
        out = outputs.logits

        out_rearranged = einops.rearrange(out, "b n d -> (b n) d")
        target = einops.rearrange(tgt, "b n -> (b n)")
        loss = outputs.loss

        total_loss += loss.item()
        total_tokens += n_tokens
        tokens += n_tokens
        total_dist += calculate_average_distance(out_rearranged, target, pad_idx=pad_idx)

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader), total_dist / len(dataloader)
