import time
from typing import Iterable

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

import wandb
from data.dataset import MyTokenizedMidiDataset
from utils import calculate_accuracy, calculate_average_distance


def train_model(
    model: nn.Module,
    train_dataset: MyTokenizedMidiDataset,
    val_dataset: MyTokenizedMidiDataset,
    cfg: DictConfig,
) -> nn.Module:
    model.to(cfg.device)
    pad_idx = train_dataset.encoder.token_to_id["<PAD>"]
    cls_idx = train_dataset.encoder.token_to_id["<CLS>"]

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

    best_test_loss = float("inf")
    for epoch in range(cfg.train.num_epochs):
        model.train()
        print(f"Epoch {epoch}", flush=True)

        # Train model for one epoch
        t_loss, t_dist, t_acc = train_epoch(
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optimizer=optimizer,
            pad_idx=pad_idx,
            cls_idx=cls_idx,
            accum_iter=cfg.train.accum_iter,
            log=cfg.log,
            log_frequency=cfg.log_frequency,
            device=cfg.device,
        )

        print(f"Epoch {epoch} Validation", flush=True)
        model.eval()
        # Evaluate the model on validation set
        v_loss, v_dist, v_acc = val_epoch(
            dataloader=val_dataloader,
            model=model,
            device=cfg.device,
            pad_idx=pad_idx,
            cls_idx=cls_idx,
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
                    "val/accuracy_epoch": v_acc,
                    "train/loss_epoch": t_loss,
                    "train/dist_epoch": t_dist,
                    "train/accuracy_epoch": t_acc,
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
    val_dataloader: Iterable,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    pad_idx: int = 1,
    cls_idx: int = 0,
    accum_iter: int = 1,
    log_frequency: int = 10,
    log: bool = False,
    device: str = "cpu",
) -> tuple[float, float, float]:
    start: float = time.time()

    total_loss: float = 0
    total_dist: float = 0
    total_acc: float = 0

    tokens: int = 0
    n_accum: int = 0
    it: int = 0

    # create progress bar
    steps: int = len(dataloader)
    progress_bar = tqdm(dataloader, total=steps)
    for batch in progress_bar:
        src: torch.Tensor = batch["source_token_ids"].to(device)
        tgt: torch.Tensor = batch["target_token_ids"].to(device)
        n_tokens: int = tgt.numel()
        tgt[tgt == pad_idx] = -100
        tgt[tgt == cls_idx] = -100
        attention_mask = src != pad_idx

        outputs = model(
            input_ids=src,
            attention_mask=attention_mask,
            labels=tgt,
        )
        out = outputs.logits

        out_rearranged = torch.reshape(out, [out.size(0) * out.size(1), out.size(-1)])
        target = torch.reshape(tgt, [tgt.size(0) * tgt.size(1)])

        loss = outputs.loss
        loss.backward()

        dist = calculate_average_distance(out_rearranged, target, pad_idx=pad_idx)
        accuracy = calculate_accuracy(out_rearranged, target, pad_idx=pad_idx)

        # Update the model parameters and optimizer gradients every `accum_iter` iterations
        if it % accum_iter == 0 or it == steps - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
        it += 1

        # Update loss and token counts
        loss_item = loss.item()
        total_loss += loss.item()
        tokens += n_tokens
        total_dist += dist
        total_acc += accuracy

        # log metrics every log_frequency steps
        if it % log_frequency == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            tok_rate = tokens / elapsed
            progress_bar.set_description(
                f"Step: {it:6d}/{steps} | acc_step: {n_accum:3d} | loss: {loss_item:6.2f} | dist: {dist:6.2f} "
                + f"| acc: {accuracy:6.2f} | tps: {tok_rate:7.1f} | lr: {lr:6.1e}"
            )

            # log the loss each to Weights and Biases
            if log:
                wandb.log({"train/loss_step": loss.item(), "train/dist_step": dist, "train/accuracy_step": accuracy})

        if it % log_frequency * 200 == 1:
            val_loss, val_dist, val_acc = val_epoch(
                dataloader=val_dataloader,
                model=model,
                pad_idx=pad_idx,
                cls_idx=cls_idx,
                device=device,
            )
            if log:
                wandb.log({"val/loss_step": val_loss, "val/dist_step": val_dist, "val/accuracy_step": val_acc})

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader), total_dist / len(dataloader), total_acc / len(dataloader)


@torch.no_grad()
def val_epoch(
    dataloader: Iterable,
    model: nn.Module,
    pad_idx: int = 1,
    cls_idx: int = 0,
    device: str = "cpu",
) -> tuple[float, float, float]:
    total_tokens: int = 0
    total_loss: float = 0
    total_acc: float = 0
    tokens: int = 0
    total_dist: float = 0

    for batch in tqdm(dataloader):
        src: torch.Tensor = batch["source_token_ids"].to(device)
        tgt: torch.Tensor = batch["target_token_ids"].to(device)
        n_tokens: int = tgt.numel()
        tgt[tgt == pad_idx] = -100
        tgt[tgt == cls_idx] = -100
        attention_mask = src != pad_idx

        outputs = model(
            input_ids=src,
            attention_mask=attention_mask,
            labels=tgt,
        )
        out = outputs.logits

        out_rearranged = torch.reshape(out, [out.size(0) * out.size(1), out.size(-1)])
        target = torch.reshape(tgt, [tgt.size(0) * tgt.size(1)])
        loss = outputs.loss

        total_loss += loss.item()
        total_tokens += n_tokens
        tokens += n_tokens
        total_dist += calculate_average_distance(out_rearranged, target, pad_idx=pad_idx)
        total_acc += calculate_accuracy(out_rearranged, target, pad_idx=pad_idx)

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader), total_dist / len(dataloader), total_acc / len(dataloader)
