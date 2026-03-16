import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

from utils import (
    set_seed, 
    list_npz_files, 
    split_trials, 
    ordinal_targets, 
    ordinal_bce_loss, 
    ordinal_decode, 
    confusion_matrix_3
)
from models import TactileBackboneTCN, ComplianceHead, SlipOrdinalHead
from datasets import ComplianceDataset, SlipOrdinalDataset

# -----------------------------
# Train / Eval Loops
# -----------------------------
def train_cls_epoch(backbone, head, loader, optimizer, device, epoch: int, desc: str, class_weights: Optional[torch.Tensor] = None):
    backbone.train()
    head.train()
    total = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"{desc} Train E{epoch}", leave=False, dynamic_ncols=True)
    for x, y, _trial_id in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = backbone(x)
        logits = head(z)
        loss = F.cross_entropy(logits, y, weight=class_weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total / max(1, n)

@torch.no_grad()
def eval_cls_epoch(backbone, head, loader, device, epoch: int, desc: str, class_weights: Optional[torch.Tensor] = None):
    backbone.eval()
    head.eval()
    total = 0.0
    correct = 0
    n = 0

    pbar = tqdm(loader, desc=f"{desc} Val   E{epoch}", leave=False, dynamic_ncols=True)
    for x, y, _trial_id in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = backbone(x)
        logits = head(z)
        loss = F.cross_entropy(logits, y, weight=class_weights)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct / max(1, n)):.3f}")

    return total / max(1, n), correct / max(1, n)

def train_ordinal_epoch(backbone, head, loader, optimizer, device, epoch: int, desc: str, pos_weight: Optional[torch.Tensor] = None):
    backbone.train()
    head.train()
    total = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"{desc} Train E{epoch}", leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = backbone(x)
        logits = head(z)
        loss = ordinal_bce_loss(logits, y, pos_weight=pos_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total / max(1, n)

@torch.no_grad()
def eval_ordinal_epoch(backbone, head, loader, device, epoch: int, desc: str, pos_weight: Optional[torch.Tensor] = None):
    backbone.eval()
    head.eval()
    total = 0.0
    n = 0
    ys_true = []
    ys_pred = []

    pbar = tqdm(loader, desc=f"{desc} Val   E{epoch}", leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = backbone(x)
        logits = head(z)
        loss = ordinal_bce_loss(logits, y, pos_weight=pos_weight)
        pred = ordinal_decode(logits)

        ys_true.append(y.detach().cpu().numpy())
        ys_pred.append(pred.detach().cpu().numpy())

        bs = x.size(0)
        total += loss.item() * bs
        n += bs

    y_true = np.concatenate(ys_true, axis=0) if len(ys_true) else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(ys_pred, axis=0) if len(ys_pred) else np.zeros((0,), dtype=np.int64)

    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0
    mae = float(np.abs(y_true - y_pred).mean()) if y_true.size > 0 else 0.0
    cm = confusion_matrix_3(y_true, y_pred) if y_true.size > 0 else np.zeros((3, 3), dtype=np.int64)

    return total / max(1, n), acc, mae, cm

# -----------------------------
# Checkpoint Loading
# -----------------------------
def load_pretrained_backbone(backbone: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    print(f"[Load] missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")

# -----------------------------
# Main Execution
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--task", default="both", choices=["compliance", "slip", "both"])
    ap.add_argument("--out-dir", default="runs_finetune")

    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.15)

    # Compliance-specific arguments
    ap.add_argument("--comp-topk-peaks", type=int, default=3)
    ap.add_argument("--comp-peak-margin", type=int, default=16)
    ap.add_argument("--comp-peak-smooth-k", type=int, default=5)
    ap.add_argument("--comp-peak-jitter", type=int, default=4)
    ap.add_argument("--comp-jitter-stride", type=int, default=2)

    # Slip-specific arguments
    ap.add_argument("--bins-joint", type=int, default=20)
    ap.add_argument("--bins-scalar", type=int, default=50)
    ap.add_argument("--slip-smooth-k", type=int, default=5)
    ap.add_argument("--slip-q1", type=float, default=0.50)
    ap.add_argument("--slip-q2", type=float, default=0.85)
    ap.add_argument("--slip-min-gap", type=float, default=0.50)
    ap.add_argument("--slip-max-baseline-windows", type=int, default=16)
    ap.add_argument("--slip-tail-k", type=int, default=5)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    files = list_npz_files(args.data_dir)
    train_files, val_files = split_trials(files, args.val_ratio, args.seed)
    print(f"[Split] train_trials={len(train_files)} val_trials={len(val_files)}")

    backbone = TactileBackboneTCN(D=args.D, H=9, W=7, act="relu", use_decoder=False)
    load_pretrained_backbone(backbone, args.pretrained)
    backbone = backbone.to(args.device)

    def set_backbone_trainable(trainable: bool):
        for p in backbone.parameters():
            p.requires_grad = trainable

    def build_optimizer(bb: nn.Module, head: nn.Module):
        bb_params = [p for p in bb.parameters() if p.requires_grad] 
        param_groups = []
        if len(bb_params) > 0:
            param_groups.append({"params": bb_params, "lr": args.lr_backbone})
        param_groups.append({"params": head.parameters(), "lr": args.lr_head})
        return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # -------------------------
    # Compliance Fine-tune
    # -------------------------
    if args.task in ["compliance", "both"]:
        print("\n=== Fine-tune: COMPLIANCE head (peak-aware Soft/Hard classification) ===")

        comp_train = ComplianceDataset(
            train_files, win=args.win, stride=args.stride,
            topk_peaks=args.comp_topk_peaks, peak_margin=args.comp_peak_margin,
            peak_smooth_k=args.comp_peak_smooth_k, peak_jitter=args.comp_peak_jitter,
            jitter_stride=args.comp_jitter_stride
        )
        comp_val = ComplianceDataset(
            val_files, win=args.win, stride=args.stride,
            topk_peaks=args.comp_topk_peaks, peak_margin=args.comp_peak_margin,
            peak_smooth_k=args.comp_peak_smooth_k, peak_jitter=args.comp_peak_jitter,
            jitter_stride=args.comp_jitter_stride
        )

        train_loader = DataLoader(
            comp_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            comp_val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False
        )

        ys_train = [y for (_, _, y, _) in comp_train.items]
        soft_n = sum(1 for y in ys_train if y == 0)
        hard_n = sum(1 for y in ys_train if y == 1)
        total_n = soft_n + hard_n
        w_soft = total_n / max(1, 2 * soft_n)
        w_hard = total_n / max(1, 2 * hard_n)
        class_weights = torch.tensor([w_soft, w_hard], dtype=torch.float32, device=args.device)

        print(f"[COMP] train class count: soft={soft_n}, hard={hard_n}, weights={class_weights.tolist()}")

        head = ComplianceHead(args.D).to(args.device)
        best_val = float("inf")

        for ep in range(1, args.epochs + 1):
            set_backbone_trainable(ep > args.freeze_backbone_epochs)
            optim = build_optimizer(backbone, head)

            tr = train_cls_epoch(backbone, head, train_loader, optim, args.device, ep, "COMP", class_weights)
            va, acc = eval_cls_epoch(backbone, head, val_loader, args.device, ep, "COMP", class_weights)

            print(f"[COMP Epoch {ep:03d}] train_loss={tr:.6f}  val_loss={va:.6f}  val_acc={acc:.4f}")
            
            # Write to TensorBoard
            writer.add_scalar("COMP/Loss_Train", tr, ep)
            writer.add_scalar("COMP/Loss_Val", va, ep)
            writer.add_scalar("COMP/Metric_Val_Acc", acc, ep)

            ckpt = {
                "epoch": ep,
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "args": vars(args),
                "task": "compliance",
            }
            torch.save(ckpt, os.path.join(args.out_dir, "comp_last.pt"))

            if va < best_val:
                best_val = va
                torch.save(ckpt, os.path.join(args.out_dir, "comp_best.pt"))

    # -------------------------
    # Slip Ordinal Fine-tune
    # -------------------------
    if args.task in ["slip", "both"]:
        print("\n=== Fine-tune: SLIP head (ordinal stable / pre-slip / slip) ===")

        slip_train = SlipOrdinalDataset(
            train_files, win=args.win, stride=args.stride,
            bins_joint=args.bins_joint, bins_scalar=args.bins_scalar,
            smooth_k=args.slip_smooth_k, q1=args.slip_q1, q2=args.slip_q2,
            min_score_gap=args.slip_min_gap, max_baseline_windows=args.slip_max_baseline_windows
        )
        slip_val = SlipOrdinalDataset(
            val_files, win=args.win, stride=args.stride,
            bins_joint=args.bins_joint, bins_scalar=args.bins_scalar,
            smooth_k=args.slip_smooth_k, q1=args.slip_q1, q2=args.slip_q2,
            min_score_gap=args.slip_min_gap, max_baseline_windows=args.slip_max_baseline_windows,
            global_thresholds=(slip_train.lo, slip_train.hi) 
        )

        train_loader = DataLoader(
            slip_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            slip_val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False
        )

        ys = [y for (_, _, y) in slip_train.items]
        ord_t = ordinal_targets(torch.tensor(ys, dtype=torch.long), num_levels=3)
        pos_counts = ord_t.sum(dim=0)
        neg_counts = ord_t.shape[0] - pos_counts
        pos_weight = (neg_counts / torch.clamp(pos_counts, min=1.0)).to(dtype=torch.float32, device=args.device)

        print(
            f"[SLIP] train class count: stable={sum(y == 0 for y in ys)}, "
            f"pre_slip={sum(y == 1 for y in ys)}, "
            f"slip={sum(y == 2 for y in ys)}, "
            f"pos_weight={pos_weight.tolist()}"
        )

        head = SlipOrdinalHead(args.D, tail_k=args.slip_tail_k).to(args.device)

        best_val_loss = float("inf")
        best_val_acc = -1.0
        best_val_mae = float("inf")

        for ep in range(1, args.epochs + 1):
            set_backbone_trainable(ep > args.freeze_backbone_epochs)
            optim = build_optimizer(backbone, head)

            tr = train_ordinal_epoch(
                backbone, head, train_loader, optim, args.device, ep, "SLIP", pos_weight=pos_weight
            )
            va, acc, mae, cm = eval_ordinal_epoch(
                backbone, head, val_loader, args.device, ep, "SLIP", pos_weight=pos_weight
            )

            print(
                f"[SLIP Epoch {ep:03d}] train_loss={tr:.6f}  "
                f"val_loss={va:.6f}  val_acc={acc:.4f}  val_mae={mae:.4f}"
            )
            print("[SLIP Confusion Matrix]\n", cm)
            
            # Write to TensorBoard
            writer.add_scalar("SLIP/Loss_Train", tr, ep)
            writer.add_scalar("SLIP/Loss_Val", va, ep)
            writer.add_scalar("SLIP/Metric_Val_Acc", acc, ep)
            writer.add_scalar("SLIP/Metric_Val_MAE", mae, ep)

            ckpt = {
                "epoch": ep,
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "pos_weight": pos_weight.detach().cpu().tolist(),
                "args": vars(args),
                "task": "slip_ordinal",
                "val_loss": va,
                "val_acc": acc,
                "val_mae": mae,
                "cm": cm.tolist(),
            }

            torch.save(ckpt, os.path.join(args.out_dir, "slip_last.pt"))

            if va < best_val_loss:
                best_val_loss = va
                torch.save(ckpt, os.path.join(args.out_dir, "slip_best_loss.pt"))

            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(ckpt, os.path.join(args.out_dir, "slip_best_acc.pt"))

            if mae < best_val_mae:
                best_val_mae = mae
                torch.save(ckpt, os.path.join(args.out_dir, "slip_best_mae.pt"))

    print("\nDone. Checkpoints saved to:", args.out_dir)
    writer.close()

if __name__ == "__main__":
    main()