# src/evaluate.py
"""
Unified evaluation script with ALL performance metrics.

✔ Base-paper metrics (UNCHANGED computation)
✔ Extended metrics added safely
✔ Windows-safe (num_workers = 0)
✔ Single evaluation pass
✔ Clean output under one heading

Metrics reported:
1. IoU
2. Precision
3. Recall
4. F1-score
5. F2-score
6. MAE
7. PR-AUC
8. MCC
9. Pixel Accuracy
10. Boundary F1 (BF1)
11. Boundary IoU (BIoU)
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef
from scipy.ndimage import binary_dilation, binary_erosion

from data.casia2_dataset import CASIA2Dataset
from models.full_model import FullModel


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def compute_prf_iou(pred, gt, eps=1e-8):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return precision, recall, f1, iou, acc


def sweep_best_metrics(prob, gt, thresholds):
    best_f1 = -1.0
    best = None
    for t in thresholds:
        pred = (prob >= t).astype(np.uint8)
        p, r, f1, iou, _ = compute_prf_iou(pred, gt)
        if f1 > best_f1:
            best_f1 = f1
            best = (p, r, f1, iou, t)
    return best


def extract_boundary(mask, dilation=2):
    mask = mask.astype(bool)
    dil = binary_dilation(mask, iterations=dilation)
    ero = binary_erosion(mask, iterations=dilation)
    return (dil ^ ero).astype(np.uint8)


# --------------------------------------------------
# Args
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval-split", choices=["val", "full"], default="val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--thresh-steps", type=int, default=200)
    return p.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = CASIA2Dataset(args.data_root, input_size=512, train=False)

    if args.eval_split == "val":
        n = len(dataset)
        val_n = int(0.1 * n)
        _, dataset = random_split(
            dataset,
            [n - val_n, val_n],
            generator=torch.Generator().manual_seed(42)
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0  # 🔥 WINDOWS SAFE
    )

    model = FullModel(pretrained_backbone=True).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    thresholds = np.linspace(0, 1, args.thresh_steps)

    # Accumulators
    P, R, F1, IoU, ACC = [], [], [], [], []
    bf1s, bious = [], []
    all_probs, all_gts, all_preds = [], [], []

    with torch.no_grad():
        for sample in tqdm(loader, desc="Evaluating"):
            img = sample["image"].to(device)
            gt = sample["mask"].squeeze().cpu().numpy().astype(np.uint8)

            out = model(img, return_all=True)
            prob = torch.sigmoid(out["mask_logits_up"]).squeeze().cpu().numpy()

            # ---- Base-paper best threshold (UNCHANGED)
            p, r, f1, iou, _ = sweep_best_metrics(prob, gt, thresholds)
            P.append(p)
            R.append(r)
            F1.append(f1)
            IoU.append(iou)

            # ---- Fixed threshold metrics
            pred = (prob >= 0.5).astype(np.uint8)
            _, _, _, _, acc = compute_prf_iou(pred, gt)
            ACC.append(acc)

            all_probs.append(prob.flatten())
            all_gts.append(gt.flatten())
            all_preds.append(pred.flatten())

            pb = extract_boundary(pred)
            gb = extract_boundary(gt)

            tp = np.sum((pb == 1) & (gb == 1))
            fp = np.sum((pb == 1) & (gb == 0))
            fn = np.sum((pb == 0) & (gb == 1))

            bf1s.append(2 * tp / (2 * tp + fp + fn + 1e-8))
            bious.append(tp / (tp + fp + fn + 1e-8))

    # Flatten for global metrics
    all_probs = np.concatenate(all_probs)
    all_gts = np.concatenate(all_gts)
    all_preds = np.concatenate(all_preds)

    MAE = np.mean(np.abs(all_probs - all_gts))
    prec_curve, rec_curve, _ = precision_recall_curve(all_gts, all_probs)
    PRAUC = auc(rec_curve, prec_curve)
    MCC = matthews_corrcoef(all_gts, all_preds)

    TP = np.sum((all_preds == 1) & (all_gts == 1))
    FP = np.sum((all_preds == 1) & (all_gts == 0))
    FN = np.sum((all_preds == 0) & (all_gts == 1))
    Precision = TP / (TP + FP + 1e-8)
    Recall = TP / (TP + FN + 1e-8)
    F2 = (5 * Precision * Recall) / (4 * Precision + Recall + 1e-8)

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------
    print("\n========== PERFORMANCE METRICS ==========")
    print("Precision        :", np.mean(P))
    print("Recall           :", np.mean(R))
    print("F1-score         :", np.mean(F1))
    print("F2-score         :", F2)
    print("IoU              :", np.mean(IoU))
    print("Pixel Accuracy   :", np.mean(ACC))
    print("MAE              :", MAE)
    print("PR-AUC           :", PRAUC)
    print("MCC              :", MCC)
    print("Boundary F1      :", np.mean(bf1s))
    print("Boundary IoU     :", np.mean(bious))
    print("========================================\n")
    print("✅ Evaluation completed successfully")


if __name__ == "__main__":
    main()
