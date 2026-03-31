import os
import argparse
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from dataset import (
    ImageDataset,
    get_train_transform,
    get_val_transform,
    make_balanced_sampler,
    mixup_data,
    cutmix_data,
    mixup_criterion,
)
from model import build_model, count_parameters
from utils import (
    SAM,
    LabelSmoothingCrossEntropy,
    ModelEMA,
    CosineAnnealingWarmup,
    AverageMeter,
    accuracy,
)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset sanity check
# ---------------------------------------------------------------------------

def check_dataset(data_root: str):
    print("\n[Dataset Check]")
    for split in ("train", "val"):
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"  WARNING: {split_dir} not found!")
            continue
        class_dirs = [
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ]
        total_imgs = sum(
            len([f for f in os.listdir(os.path.join(split_dir, c))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            for c in class_dirs
        )
        print(f"  {split:5s}: {len(class_dirs)} classes, {total_imgs} images")
        if class_dirs:
            sample = sorted(class_dirs, key=lambda x: int(x) if x.isdigit() else x)
            print(f"           class names (first 5): {sample[:5]}")
    print()


# ---------------------------------------------------------------------------
# Training: AdamW + AMP (stable, fast)
# ---------------------------------------------------------------------------

def train_adamw(model, loader, optimizer, criterion, device, ema,
                scaler, mixup_prob, clip_grad):
    model.train()
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        r = random.random()
        if r < mixup_prob * 0.5:
            images, la, lb, lam = mixup_data(images, labels, alpha=0.4)
            mixed = True
        elif r < mixup_prob:
            images, la, lb, lam = cutmix_data(images, labels, alpha=1.0)
            mixed = True
        else:
            mixed = False

        optimizer.zero_grad()
        with autocast():
            out = model(images)
            loss = (mixup_criterion(criterion, out, la, lb, lam)
                    if mixed else criterion(out, labels))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        top1, top5 = accuracy(out, labels)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(top1, bs)
        top5_m.update(top5, bs)

    return loss_m.avg, top1_m.avg, top5_m.avg


# ---------------------------------------------------------------------------
# Training: SAM (no AMP — avoids scaler conflict)
# ---------------------------------------------------------------------------

def train_sam(model, loader, optimizer, criterion, device, ema,
              mixup_prob, clip_grad):
    model.train()
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        r = random.random()
        if r < mixup_prob * 0.5:
            images, la, lb, lam = mixup_data(images, labels, alpha=0.4)
            mixed = True
        elif r < mixup_prob:
            images, la, lb, lam = cutmix_data(images, labels, alpha=1.0)
            mixed = True
        else:
            mixed = False

        out = model(images)
        loss = (mixup_criterion(criterion, out, la, lb, lam)
                if mixed else criterion(out, labels))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.first_step(zero_grad=True)

        out2 = model(images)
        loss2 = (mixup_criterion(criterion, out2, la, lb, lam)
                 if mixed else criterion(out2, labels))
        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.second_step(zero_grad=True)

        if ema is not None:
            ema.update(model)

        top1, top5 = accuracy(out, labels)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(top1, bs)
        top5_m.update(top5, bs)

    return loss_m.avg, top1_m.avg, top5_m.avg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images)
        loss = criterion(out, labels)
        top1, top5 = accuracy(out, labels)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(top1, bs)
        top5_m.update(top5, bs)

    return loss_m.avg, top1_m.avg, top5_m.avg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--model", default="resnet152",
                   choices=["resnet50", "resnet101", "resnet152"])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--mixup_prob", type=float, default=0.5)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--drop_path", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--use_sam", action="store_true",
                   help="SAM optimizer (no AMP, ~2x slower but better generalization)")
    p.add_argument("--balanced_sampler", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_grad", type=float, default=5.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    check_dataset(args.data_root)

    train_tf = get_train_transform(args.img_size)
    val_tf = get_val_transform(args.img_size)

    train_ds = ImageDataset(os.path.join(args.data_root, "train"), train_tf)
    val_ds = ImageDataset(os.path.join(args.data_root, "val"), val_tf)

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training images found under {args.data_root}/train/\n"
            "Expected structure: data/train/<class_id>/<image>.jpg"
        )

    sampler = make_balanced_sampler(train_ds) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sample_imgs, sample_lbls = next(iter(train_loader))
    print(f"[Batch check] images: {sample_imgs.shape}  labels: {sample_lbls[:8].tolist()}")
    print(f"              label range: {sample_lbls.min().item()} ~ {sample_lbls.max().item()}\n")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = build_model(
        args.model,
        num_classes=100,
        pretrained=True,
        drop_path_rate=args.drop_path,
        dropout=args.dropout,
    ).to(device)

    print(f"Model       : {args.model} + CBAM + Multi-scale GeM")
    print(f"Params      : {count_parameters(model) / 1e6:.2f}M")
    print(f"Optimizer   : {'SAM (no AMP)' if args.use_sam else 'AdamW + AMP'}")

    ema = ModelEMA(model, decay=args.ema_decay)
    criterion = LabelSmoothingCrossEntropy(args.label_smooth)

    # -----------------------------------------------------------------------
    # Optimizer & Scheduler
    # -----------------------------------------------------------------------
    if args.use_sam:
        optimizer = SAM(
            model.parameters(),
            base_optimizer=torch.optim.AdamW,
            rho=0.05,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        base_opt = optimizer.base_optimizer
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        base_opt = optimizer

    scheduler = CosineAnnealingWarmup(
        base_opt,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        min_lr=1e-6,
    )
    scaler = GradScaler()

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        print(f"Resuming    : {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("ema", ckpt.get("model", ckpt))
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:3]} ...")
        ema = ModelEMA(model, decay=args.ema_decay)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    header = (f"{'Ep':>4}  {'LR':>9}  {'TrLoss':>8}  {'TrTop1':>7}  "
              f"{'ValLoss':>8}  {'ValTop1':>7}  {'ValTop5':>7}  {'Best':>7}")
    print("\n" + "=" * 78)
    print(header)
    print("=" * 78)
    
    history = []
    
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        if args.use_sam:
            tr_loss, tr_top1, _ = train_sam(
                model, train_loader, optimizer, criterion, device, ema,
                args.mixup_prob, clip_grad=args.clip_grad,
            )
        else:
            tr_loss, tr_top1, _ = train_adamw(
                model, train_loader, optimizer, criterion, device, ema,
                scaler, args.mixup_prob, clip_grad=args.clip_grad,
            )

        val_loss, val_top1, val_top5 = validate(
            ema.module, val_loader, criterion, device
        )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        is_best = val_top1 > best_acc
        marker = " <- best" if is_best else ""
        print(
            f"{epoch+1:>4d}  {lr:>9.2e}  {tr_loss:>8.4f}  {tr_top1:>6.2f}%"
            f"  {val_loss:>8.4f}  {val_top1:>6.2f}%  {val_top5:>6.2f}%"
            f"  {best_acc:>6.2f}%  ({elapsed:.0f}s){marker}"
        )

        if is_best:
            best_acc = val_top1
            torch.save(
                {
                    "epoch": epoch,
                    "img_size": args.img_size,
                    "model": model.state_dict(),
                    "ema": ema.module.state_dict(),
                    "best_acc": best_acc,
                },
                os.path.join(args.save_dir, f"best_{args.model}.pth"),
            )

        torch.save(
            {
                "epoch": epoch,
                "img_size": args.img_size,
                "model": model.state_dict(),
                "ema": ema.module.state_dict(),
                "optimizer": base_opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            },
            os.path.join(args.save_dir, f"latest_{args.model}.pth"),
        )
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': tr_loss,
            'val_acc': val_top1
        })
    
    log_filename = os.path.join(args.save_dir, f"train_log_{args.img_size}.csv")
    pd.DataFrame(history).to_csv(log_filename, index=False)

    print("=" * 78)
    print(f"Done.  Best val Top-1 = {best_acc:.2f}%")
    print(f"Saved -> {os.path.join(args.save_dir, f'best_{args.model}.pth')}")


if __name__ == "__main__":
    main()