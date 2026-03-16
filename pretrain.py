import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import from your custom modules
from utils import set_seed, make_mask, augment_tactile_batch
from models import TactileBackboneTCN
from datasets import TactileWindowDataset

# -----------------------------
# Train and Evaluation Loops
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, mask_ratio: float, lam_e: float, lam_d: float,
                    epoch: int = 0, act_hi_pctl: float = 90.0, act_eps: float = 1e-6):
    model.train()
    total_loss = 0.0
    n = 0
    hi_loss_sum = 0.0
    lo_loss_sum = 0.0
    hi_cnt = 0
    lo_cnt = 0

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False, dynamic_ncols=True)
    for x in pbar:
        x = x.to(device, non_blocking=True)  # (B, L, 3, H, W)
        x = augment_tactile_batch(x, p=0.5)  # Apply data augmentation
        B, L, C, H, W = x.shape

        m = make_mask(L, H, W, mask_ratio).to(device)   # (L, H, W)
        mm = m.unsqueeze(0).expand(B, L, H, W)          # (B, L, H, W)
        m_ch = m.unsqueeze(0).unsqueeze(2).expand(B, L, C, H, W)

        x_in = x.clone()
        x_in[m_ch] = 0.0

        # Forward pass (Pretrain mode returns features and reconstructed x_hat)
        _, x_hat = model(x_in)

        # Original masked reconstruction losses
        ev_loss = F.smooth_l1_loss(x_hat[:, :, 0][mm], x[:, :, 0][mm], reduction="mean")
        dx_loss = F.smooth_l1_loss(x_hat[:, :, 1][mm], x[:, :, 1][mm], reduction="mean")
        dy_loss = F.smooth_l1_loss(x_hat[:, :, 2][mm], x[:, :, 2][mm], reduction="mean")
        loss = lam_e * ev_loss + lam_d * (dx_loss + dy_loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        n += B

        # Activity split on masked positions
        # channel0 = log1p(ev), channel1/2 = dx/dy
        # act = ev + sqrt(dx^2 + dy^2)
        ev_t = x[:, :, 0]
        dx_t = x[:, :, 1]
        dy_t = x[:, :, 2]
        act = ev_t + torch.sqrt(dx_t * dx_t + dy_t * dy_t + 1e-12)

        act_m = act[mm]  # 1D over masked elements
        if act_m.numel() > 0:
            thr = torch.quantile(act_m.detach(), act_hi_pctl / 100.0)
            thr = torch.maximum(thr, torch.tensor(act_eps, device=device))
            hi = act >= thr
            lo = act < thr

            hi_mm = mm & hi
            lo_mm = mm & lo

            # Compute per-element smooth_l1 then average over hi/lo masked areas
            def split_loss(ch_idx: int, mask_bool: torch.Tensor):
                if mask_bool.any():
                    diff = F.smooth_l1_loss(x_hat[:, :, ch_idx][mask_bool],
                                            x[:, :, ch_idx][mask_bool],
                                            reduction="mean")
                    return diff
                return None

            hi_ev = split_loss(0, hi_mm)
            hi_dx = split_loss(1, hi_mm)
            hi_dy = split_loss(2, hi_mm)
            lo_ev = split_loss(0, lo_mm)
            lo_dx = split_loss(1, lo_mm)
            lo_dy = split_loss(2, lo_mm)

            if hi_ev is not None:
                hi_total = lam_e * hi_ev + lam_d * (hi_dx + hi_dy)
                hi_loss_sum += float(hi_total.detach().cpu()) * B
                hi_cnt += B
            if lo_ev is not None:
                lo_total = lam_e * lo_ev + lam_d * (lo_dx + lo_dy)
                lo_loss_sum += float(lo_total.detach().cpu()) * B
                lo_cnt += B

            pbar.set_postfix(loss=f"{loss.item():.5f}",
                             hi=f"{(hi_loss_sum/max(1,hi_cnt)):.4f}",
                             lo=f"{(lo_loss_sum/max(1,lo_cnt)):.4f}",
                             thr=f"{float(thr.detach().cpu()):.4f}")
        else:
            pbar.set_postfix(loss=f"{loss.item():.5f}")

    avg_loss = total_loss / max(1, n)
    hi_avg = hi_loss_sum / max(1, hi_cnt)
    lo_avg = lo_loss_sum / max(1, lo_cnt)
    return avg_loss, hi_avg, lo_avg


@torch.no_grad()
def eval_one_epoch(model, loader, device, mask_ratio: float, lam_e: float, lam_d: float, epoch: int = 0):
    model.eval()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Val", leave=False)
    for x in pbar:
        x = x.to(device)
        B, L, C, H, W = x.shape

        m = make_mask(L, H, W, mask_ratio).to(device)
        m_ch = m.unsqueeze(0).unsqueeze(2).expand(B, L, C, H, W)

        x_in = x.clone()
        x_in[m_ch] = 0.0

        _, x_hat = model(x_in)

        ev_loss = F.smooth_l1_loss(x_hat[:, :, 0][m.unsqueeze(0).expand(B, L, H, W)],
                                   x[:, :, 0][m.unsqueeze(0).expand(B, L, H, W)],
                                   reduction="mean")
        dx_loss = F.smooth_l1_loss(x_hat[:, :, 1][m.unsqueeze(0).expand(B, L, H, W)],
                                   x[:, :, 1][m.unsqueeze(0).expand(B, L, H, W)],
                                   reduction="mean")
        dy_loss = F.smooth_l1_loss(x_hat[:, :, 2][m.unsqueeze(0).expand(B, L, H, W)],
                                   x[:, :, 2][m.unsqueeze(0).expand(B, L, H, W)],
                                   reduction="mean")
        loss = lam_e * ev_loss + lam_d * (dx_loss + dy_loss)

        total_loss += loss.item() * B
        n += B

    return total_loss / max(1, n)


# -----------------------------
# Main Execution
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder containing .npz files")
    ap.add_argument("--out-dir", default="runs_pretrain_tcn_v2", help="Output folder")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.30)
    ap.add_argument("--D", type=int, default=16, help="Backbone channels (keep small for <1ms inference)")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lam-e", type=float, default=1.0, help="Loss weight for event channel")
    ap.add_argument("--lam-d", type=float, default=1.0, help="Loss weight for dx/dy channels")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phases", default="", help='e.g. "RECORDING_TEXTURE,COLLECTING_SLIP"')
    ap.add_argument("--max-files", type=int, default=0)

    ap.add_argument("--act-hi-pctl", type=float, default=90.0, help="Percentile for high activity split")
    ap.add_argument("--act-eps", type=float, default=1e-6, help="Epsilon for activity threshold")

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    phases = None
    if args.phases.strip():
        phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    max_files = args.max_files if args.max_files > 0 else None

    # Load Dataset
    ds = TactileWindowDataset(data_dir=args.data_dir, win=args.win, stride=args.stride, phases=phases, H=9, W=7, max_files=max_files)

    n_total = len(ds)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = TactileBackboneTCN(D=args.D, H=9, W=7, act="relu", use_decoder=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        tr, tr_hi, tr_lo = train_one_epoch(
            model, train_loader, optimizer, args.device,
            mask_ratio=args.mask_ratio, lam_e=args.lam_e, lam_d=args.lam_d,
            epoch=ep, act_hi_pctl=args.act_hi_pctl, act_eps=args.act_eps
        )
        va = eval_one_epoch(model, val_loader, args.device,
                            mask_ratio=args.mask_ratio, lam_e=args.lam_e, lam_d=args.lam_d)

        print(f"[Epoch {ep:03d}] train_loss={tr:.6f} (hi={tr_hi:.6f}, lo={tr_lo:.6f})  val_loss={va:.6f}")
        
        # Write to TensorBoard
        writer.add_scalar("Loss/Train_Total", tr, ep)
        writer.add_scalar("Loss/Train_High_Activity", tr_hi, ep)
        writer.add_scalar("Loss/Train_Low_Activity", tr_lo, ep)
        writer.add_scalar("Loss/Validation", va, ep)

        # Save latest checkpoint
        ckpt_path = os.path.join(args.out_dir, "last.pt")
        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "args": vars(args),
        }, ckpt_path)

        # Save best checkpoint
        if va < best_val:
            best_val = va
            best_path = os.path.join(args.out_dir, "best.pt")
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "best_val": best_val,
                "args": vars(args),
            }, best_path)

    print(f"Done. Best val={best_val:.6f}. Checkpoints saved to: {args.out_dir}")
    writer.close()

if __name__ == "__main__":
    main()