#!/usr/bin/env python3
"""Stage 1: Huấn luyện mô hình Chẩn đoán lỗi âm vị (MSDD / MDD).

Nhiệm vụ:
  - Head 1 (Detection): Phân loại 3 lớp lỗi âm vị (0=OK, 1=Substitution, 2=Deletion, -1=Mask).
  - Head 2 (Diagnosis): Chẩn đoán đích danh âm thay thế (39 lớp ARPA39, -1=Mask).

Được quản lý hoàn toàn bằng file cấu hình YAML/JSON (ví dụ: configs/stage1/baseline.yaml).
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vh_gopt.config import load_config_file, save_config

PHONE_NUM = 39


class MSDDDataset(Dataset):
    def __init__(self, npz_path, use_occ=False, max_len=150):
        z = np.load(npz_path, allow_pickle=False)
        self.feat = torch.tensor(z["feat"], dtype=torch.float32)  # [N, L, 80]
        self.phn = torch.tensor(z["phn"].astype(np.int64))        # [N, L]
        self.msdd_type = torch.tensor(z["msdd_type"].astype(np.int64))  # [N, L] (0, 1, 2, -1)
        self.msdd_sub = torch.tensor(z["msdd_sub"].astype(np.int64))    # [N, L] (0..38, -1)

        # Chuẩn hóa feature
        valid = self.phn >= 0
        fv = self.feat[valid]
        feat_mean = float(fv.mean()) if len(fv) else 0.0
        feat_std = max(float(fv.std()), 1e-6) if len(fv) else 1.0
        self.feat = ((self.feat - feat_mean) / feat_std) * valid.unsqueeze(-1).float()

        if use_occ and "occ" in z:
            occ = torch.tensor(z["occ"], dtype=torch.float32).unsqueeze(-1)
            self.feat = torch.cat([self.feat, occ], dim=-1)

    def __len__(self):
        return self.feat.size(0)

    def __getitem__(self, idx):
        return {
            "feat": self.feat[idx],
            "phn": self.phn[idx],
            "msdd_type": self.msdd_type[idx],
            "msdd_sub": self.msdd_sub[idx],
        }


class MSDDNet(nn.Module):
    """Mô hình phát hiện & chẩn đoán lỗi âm vị Stage 1."""
    def __init__(self, in_dim=80, embed_dim=64, depth=3, heads=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.phn_embed = nn.Embedding(PHONE_NUM + 1, embed_dim, padding_idx=PHONE_NUM)
        self.pos_embed = nn.Parameter(torch.randn(1, 150, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Head 1: Detection 3 lớp (0=OK, 1=Sub, 2=Del)
        self.det_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
        )

        # Head 2: Diagnosis 39 lớp (âm thay thế)
        self.diag_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, PHONE_NUM),
        )

    def forward(self, feat, phn):
        # phn clamp: -1 -> PHONE_NUM (padding index)
        phn_safe = torch.where(phn >= 0, phn, torch.full_like(phn, PHONE_NUM))
        x = self.in_proj(feat) + self.phn_embed(phn_safe) + self.pos_embed[:, :feat.size(1), :]

        padding_mask = (phn < 0)  # [B, L]
        h = self.encoder(x, src_key_padding_mask=padding_mask)

        det_logits = self.det_head(h)    # [B, L, 3]
        diag_logits = self.diag_head(h)  # [B, L, 39]
        return det_logits, diag_logits


def compute_metrics(det_preds, det_targets, diag_preds, diag_targets):
    """Tính các metric chuẩn cho MDD: Precision, Recall, F1, FAR, FRR, Diag Acc."""
    # Lọc mask
    valid_det = (det_targets >= 0)
    y_true = det_targets[valid_det]
    y_pred = det_preds[valid_det]

    # Detection metrics (Phát hiện âm sai: Sub=1 hoặc Del=2 vs OK=0)
    true_err = (y_true > 0).astype(int)
    pred_err = (y_pred > 0).astype(int)

    tp = np.sum((pred_err == 1) & (true_err == 1))
    fp = np.sum((pred_err == 1) & (true_err == 0))
    tn = np.sum((pred_err == 0) & (true_err == 0))
    fn = np.sum((pred_err == 0) & (true_err == 1))

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-6))
    far = float(fp / max(fp + tn, 1))   # False Alarm (chấm oan)
    frr = float(fn / max(fn + tp, 1))   # False Rejection (bỏ sót lỗi)
    acc = float((tp + tn) / max(len(true_err), 1))

    # Diagnosis metrics (Chẩn đoán đúng âm thay thế trên các mẫu golden)
    valid_diag = (diag_targets >= 0)
    diag_acc = 0.0
    if np.sum(valid_diag) > 0:
        diag_correct = np.sum(diag_preds[valid_diag] == diag_targets[valid_diag])
        diag_acc = float(diag_correct / np.sum(valid_diag))

    return {
        "det_accuracy": round(acc, 4),
        "det_precision": round(precision, 4),
        "det_recall": round(recall, 4),
        "det_f1": round(f1, 4),
        "det_far": round(far, 4),
        "det_frr": round(frr, 4),
        "diag_accuracy": round(diag_acc, 4),
        "diag_samples": int(np.sum(valid_diag)),
    }


def evaluate(model, dataloader, device):
    model.eval()
    all_det_preds, all_det_targets = [], []
    all_diag_preds, all_diag_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            feat = batch["feat"].to(device)
            phn = batch["phn"].to(device)
            det_logits, diag_logits = model(feat, phn)

            det_p = det_logits.argmax(dim=-1).cpu().numpy()
            diag_p = diag_logits.argmax(dim=-1).cpu().numpy()

            all_det_preds.append(det_p.reshape(-1))
            all_det_targets.append(batch["msdd_type"].numpy().reshape(-1))
            all_diag_preds.append(diag_p.reshape(-1))
            all_diag_targets.append(batch["msdd_sub"].numpy().reshape(-1))

    det_preds = np.concatenate(all_det_preds)
    det_targets = np.concatenate(all_det_targets)
    diag_preds = np.concatenate(all_diag_preds)
    diag_targets = np.concatenate(all_diag_targets)

    return compute_metrics(det_preds, det_targets, diag_preds, diag_targets)


def main():
    pre_p = argparse.ArgumentParser(add_help=False)
    pre_p.add_argument("--config", default=None)
    pre_args, _ = pre_p.parse_known_args()

    cfg = {}
    if pre_args.config and os.path.exists(pre_args.config):
        cfg = load_config_file(pre_args.config)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=pre_args.config, help="Path to YAML/JSON config file")
    ap.add_argument("--train", default=cfg.get("train", "data/gopt_vh_scripted_gold/train.npz"))
    ap.add_argument("--test", default=cfg.get("test", "data/gopt_vh_scripted_gold/test_unseen_speakers.npz"))
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 40))
    ap.add_argument("--bs", type=int, default=cfg.get("bs", 32))
    ap.add_argument("--lr", type=float, default=cfg.get("lr", 5e-4))
    ap.add_argument("--wd", type=float, default=cfg.get("wd", 1e-4))
    ap.add_argument("--embed-dim", type=int, default=cfg.get("embed_dim", 64))
    ap.add_argument("--depth", type=int, default=cfg.get("depth", 3))
    ap.add_argument("--heads", type=int, default=cfg.get("heads", 4))
    ap.add_argument("--dropout", type=float, default=cfg.get("dropout", 0.1))
    ap.add_argument("--w-det", type=float, default=cfg.get("w_det", 1.0))
    ap.add_argument("--w-diag", type=float, default=cfg.get("w_diag", 1.0))
    ap.add_argument("--out", default=cfg.get("out", "ckpt/stage1_msdd_baseline"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lưu lại config của thí nghiệm này
    run_cfg = vars(args).copy()
    save_config(run_cfg, out_dir / "config.yaml")

    print(f"============================================================")
    print(f"BẮT ĐẦU HUẤN LUYỆN STAGE 1 (MSDD/MDD): {args.out}")
    print(f"Config: {run_cfg}")
    print(f"============================================================")

    train_ds = MSDDDataset(args.train)
    test_ds = MSDDDataset(args.test)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False)

    in_dim = train_ds.feat.shape[-1]
    model = MSDDNet(in_dim=in_dim, embed_dim=args.embed_dim, depth=args.depth,
                    heads=args.heads, dropout=args.dropout).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Class weights cho Detection (cân bằng lớp OK vs Sub vs Del)
    det_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    diag_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_f1 = -1.0
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            feat = batch["feat"].to(args.device)
            phn = batch["phn"].to(args.device)
            t_det = batch["msdd_type"].to(args.device)
            t_diag = batch["msdd_sub"].to(args.device)

            optimizer.zero_grad()
            det_logits, diag_logits = model(feat, phn)

            l_det = det_loss_fn(det_logits.view(-1, 3), t_det.view(-1))
            l_diag = diag_loss_fn(diag_logits.view(-1, PHONE_NUM), t_diag.view(-1))
            loss = args.w_det * l_det + args.w_diag * l_diag

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            m = evaluate(model, test_loader, args.device)
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Det F1: {m['det_f1']:.4f} (P: {m['det_precision']:.4f}, R: {m['det_recall']:.4f}, FAR: {m['det_far']:.4f}) | "
                  f"Diag Acc: {m['diag_accuracy']:.4f}")

            if m["det_f1"] > best_f1:
                best_f1 = m["det_f1"]
                best_metrics = m
                torch.save(model.state_dict(), out_dir / "best_checkpoint.pt")
                with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(best_metrics, f, indent=2)

    print("\n============================================================")
    print(f"HUẤN LUYỆN STAGE 1 HOÀN TẤT!")
    print(f"Best Metrics: {json.dumps(best_metrics, indent=2)}")
    print(f"Checkpoint và config đã lưu tại: {out_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()
