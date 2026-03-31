import os
import argparse
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TestDataset, get_tta_transforms, get_val_transform
from model import build_model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_pass(model, loader, device):
    model.eval()
    all_names, all_probs = [], []
    for images, names in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_names.extend(names)
        all_probs.append(F.softmax(logits, dim=1).cpu())
    return all_names, torch.cat(all_probs, dim=0)


@torch.no_grad()
def infer_tta(model, test_root, img_size, batch_size, num_workers, device):
    tta_tfs = get_tta_transforms(img_size)
    agg_probs = None
    names = None

    for i, tf in enumerate(tta_tfs):
        ds = TestDataset(test_root, transform=tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        cur_names, cur_probs = infer_pass(model, loader, device)
        if names is None:
            names = cur_names
            agg_probs = cur_probs
        else:
            agg_probs = agg_probs + cur_probs
        print(f"    TTA {i+1}/{len(tta_tfs)} done")

    return names, agg_probs / len(tta_tfs)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HW1 Inference")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--checkpoint", nargs="+", required=True,
                   help="Checkpoint path(s). Multiple = ensemble.")
    p.add_argument("--model", nargs="+", default=["resnet152"],
                   choices=["resnet50", "resnet101", "resnet152"],
                   help="Model name(s) matching each checkpoint.")
    p.add_argument("--img_size", nargs="+", type=int, default=[320],
                   help="Image size(s) matching each checkpoint.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tta", action="store_true", help="Enable 6-way TTA")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature scaling (>1 = softer, <1 = sharper)")
    p.add_argument("--output", default="prediction.csv")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_root = os.path.join(args.data_root, "test")
    num_classes = 100

    # Broadcast single values across all checkpoints
    n = len(args.checkpoint)
    model_names = args.model if len(args.model) == n else args.model * n
    img_sizes = args.img_size if len(args.img_size) == n else args.img_size * n

    ensemble_probs = None
    names = None

    for ckpt_path, m_name, img_sz in zip(args.checkpoint, model_names, img_sizes):
        print(f"\n[{m_name}] {ckpt_path}  (img_size={img_sz})")

        # Build and load model
        model = build_model(m_name, num_classes=num_classes, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("ema", ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

        if args.tta:
            cur_names, cur_probs = infer_tta(
                model, test_root, img_sz, args.batch_size, args.num_workers, device
            )
        else:
            tf = get_val_transform(img_sz)
            ds = TestDataset(test_root, transform=tf)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            cur_names, cur_probs = infer_pass(model, loader, device)

        if args.temperature != 1.0:
            cur_probs = F.softmax(
                torch.log(cur_probs.clamp(min=1e-9)) / args.temperature, dim=1
            )

        if names is None:
            names = cur_names
            ensemble_probs = cur_probs
        else:
            ensemble_probs = ensemble_probs + cur_probs

    ensemble_probs /= n
    preds = torch.argmax(ensemble_probs, dim=1).tolist()

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        for name, pred in zip(names, preds):
            writer.writerow([name, pred])

    print(f"\nSaved {len(preds)} predictions → {args.output}")


if __name__ == "__main__":
    main()
