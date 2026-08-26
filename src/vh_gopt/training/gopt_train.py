#!/usr/bin/env python3
"""
Train GOPT on CTC-GOP 41-d features (Speechocean762) using HuggingFace Trainer.

Data handling & loss follow YuanGongND/gopt (src/traintest.py) directly:
  - GOP feature normalized to ~0-mean/unit-std with a SINGLE global scalar
    mean/std computed on the TRAIN set over valid (non-pad) tokens (norm_valid).
  - labels on 0-2 scale: phone already 0-2; word & utterance divided by 5.
  - loss = masked-mean MSE per level (phone / word / utt), summed with weights.
    (equivalent to GOPT's `loss * (B*L)/sum(mask)` rescale of a zeroed MSE.)
  - WORD PCC is evaluated at the WORD level: per-phone word-head outputs are
    averaged within each word_id, then correlated (valid_word).

GOPT is wrapped as `GOPTForScoring` returning `loss`, driven by transformers.Trainer
(wandb via report_to, tqdm, per-epoch eval, best checkpoint, HF push).

Stress (word) and prosodic (utt) heads are dropped (out of scope): word heads =
{accuracy, total}, utt heads = {accuracy, completeness, fluency, total}.

Usage:
  python gopt_train.py --epochs 80 --wandb-project gop-ctc-gopt --wandb-run gopt-41d-v1
  # add:  --push-model --hf-repo gopt-ctc-gop
  # or:   --no-wandb
"""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import Dataset

from vh_gopt.training.gopt_model import GOPT, UTT_HEADS, WORD_HEADS, MEAN_HEADS
from vh_gopt.training.gopt_hia import HIA

UTT_KEEP = [0, 1, 2, 4]   # accuracy, completeness, fluency, total (drop prosodic=3)
LABEL_NAMES = ["phone_label", "word_label", "utt_label"]
PHONE_W = 2.0             # trọng số phone trong metric 'phone_heavy' (chọn best nghiêng phone); set bởi --phone-weight


def pcc(pred, tgt, mask=None):
    p, t = np.asarray(pred, float), np.asarray(tgt, float)
    if mask is not None:
        p, t = p[mask], t[mask]
    if len(p) < 2 or p.std() == 0 or t.std() == 0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def effective_number_weights(values, n_bins=20, beta=0.999, lo=0.0, hi=2.0, wcap=5.0):
    """Score-balanced (Do et al., arXiv:2305.16664) inverse effective-number weights
    over `n_bins` score bins on [lo,hi], CLIPPED to [1/wcap, wcap] (empty bins -> 1)
    to keep near-empty bins from exploding. Normalized to mean weight 1 over samples."""
    v = np.asarray(values, float)
    v = v[(v >= lo) & (v <= hi)]
    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]                   # interior edges
    idx = np.digitize(v, edges)                                     # [n], bin per value
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    eff = np.where(counts > 0, (1 - beta) / (1 - beta ** np.maximum(counts, 1)), np.nan)
    med = np.nanmedian(eff)
    eff = np.where(np.isnan(eff), med, eff) / med                  # empty bins -> median (=1)
    eff = np.clip(eff, 1.0 / wcap, wcap)                           # tame near-empty bins
    mean_w = eff[idx].mean() if len(v) else 1.0                    # normalize to mean 1
    w = eff / max(mean_w, 1e-8)
    return edges.astype(np.float32), w.astype(np.float32)


def masked_mse_loss(out, phone_label, word_label, utt_label, w_phn=1.0, w_word=1.0, w_utt=1.0):
    """Plain masked multi-head MSE (GOPT-faithful), shared by GOPT & HIA wrappers."""
    m = (phone_label >= 0).float()
    wl = word_label[..., :len(WORD_HEADS)]
    mw = (wl >= 0).float()                 # mask WORD theo chính nhãn word (per-head), KHÔNG buộc theo phone.
    #   Trước đây mw = mask phone -> khi phone bị mask (vd kid phone=-1) thì word-loss cũng bị giết oan.
    #   SO762: token hợp lệ luôn có cả phone+word nên không đổi hành vi; kid: mở khóa tín hiệu word.
    lp = (((out["phone"] - phone_label.clamp(min=0)) ** 2) * m).sum() / m.sum().clamp_min(1)
    lw = (((out["word"] - wl.clamp(min=0)) ** 2) * mw).sum() / mw.sum().clamp_min(1)
    mu = (utt_label >= 0).float()
    lu = (((out["utt"] - utt_label) ** 2) * mu).sum() / mu.sum().clamp_min(1)
    return w_phn * lp + w_word * lw + w_utt * lu


class HIAForScoring(HIA):
    """HIA + masked multi-head MSE loss, HF-Trainer compatible."""
    def __init__(self, *a, w_phn=1.0, w_word=1.0, w_utt=1.0, noise=0.0, **k):
        super().__init__(*a, **k)
        self.w_phn, self.w_word, self.w_utt, self.noise = w_phn, w_word, w_utt, noise

    def forward(self, feat, phn, phone_label=None, word_label=None, utt_label=None):
        if self.training and self.noise > 0:
            feat = feat + (torch.rand_like(feat) - 1) * self.noise
        out = super().forward(feat, phn)
        res = {"phone": out["phone"], "word": out["word"], "utt": out["utt"]}
        if phone_label is not None:
            res["loss"] = masked_mse_loss(out, phone_label, word_label, utt_label,
                                          self.w_phn, self.w_word, self.w_utt)
        return res


class GOPTForScoring(GOPT):
    """GOPT + masked multi-head MSE loss (GOPT-faithful), HF-Trainer compatible.
    With `balanced`, MSE is reweighted per element by score-balanced weights."""
    def __init__(self, *a, w_phn=1.0, w_word=1.0, w_utt=1.0, noise=0.0,
                 balanced=False, bw_edges=None, bw_phn=None, bw_utt=None, **k):
        super().__init__(*a, **k)
        self.w_phn, self.w_word, self.w_utt, self.noise = w_phn, w_word, w_utt, noise
        self.balanced = balanced
        if balanced:
            self.register_buffer("bw_edges", torch.tensor(bw_edges))
            self.register_buffer("bw_phn", torch.tensor(bw_phn))    # for phone & word heads
            self.register_buffer("bw_utt", torch.tensor(bw_utt))

    def _w(self, label, table):
        idx = torch.bucketize(label.clamp(min=0), self.bw_edges)    # bin index (edges span nhãn thật; bucketize tự đẩy điểm cao vào bin cuối)
        return table[idx]

    def forward(self, feat, phn, phone_label=None, word_label=None, utt_label=None):
        if self.training and self.noise > 0:                       # GOPT input augmentation
            feat = feat + (torch.rand_like(feat) - 1) * self.noise
        out = super().forward(feat, phn)
        res = {"phone": out["phone"], "word": out["word"], "utt": out["utt"]}
        if phone_label is not None:
            m = (phone_label >= 0).float()                          # [B,L] valid mask (phone)
            wl = word_label[..., :len(WORD_HEADS)]                  # drop word_id channel
            mw = (wl >= 0).float()                                  # mask WORD theo nhãn word (per-head), KHÔNG buộc theo phone
            if self.balanced:
                wp = self._w(phone_label.clamp(min=0), self.bw_phn) * m
                ww = self._w(wl.clamp(min=0), self.bw_phn) * mw
                wu = self._w(utt_label, self.bw_utt)
                lp = ((out["phone"] - phone_label.clamp(min=0)) ** 2 * wp).sum() / wp.sum().clamp_min(1)
                lw = ((out["word"] - wl.clamp(min=0)) ** 2 * ww).sum() / ww.sum().clamp_min(1)
                lu = ((out["utt"] - utt_label) ** 2 * wu).sum() / wu.sum().clamp_min(1)
            else:
                lp = (((out["phone"] - phone_label.clamp(min=0)) ** 2) * m).sum() / m.sum().clamp_min(1)
                lw = (((out["word"] - wl.clamp(min=0)) ** 2) * mw).sum() / mw.sum().clamp_min(1)
                mu = (utt_label >= 0).float()
                lu = (((out["utt"] - utt_label) ** 2) * mu).sum() / mu.sum().clamp_min(1)
            res["loss"] = self.w_phn * lp + self.w_word * lw + self.w_utt * lu
        return res


class GOPTDataset(Dataset):
    """
    word_label channel layout: [accuracy/5, total/5, word_id]  (word_id kept raw,
    pad -1) so eval can aggregate to word level like GOPT's valid_word.
    """
    def __init__(self, path, feat_mean=None, feat_std=None, use_occ=False,
                 occ_mean=None, occ_std=None, use_prosody=False,
                 pros_mean=None, pros_std=None,
                 use_wavlm=False, wavlm_dim=128, wavlm_pca=None,
                 wavlm_norm=None):
        # `path` may be an npz filepath OR a preloaded dict-of-arrays (e.g. built from
        # a HuggingFace dataset split). Both support z["k"], z.get("k"), "k" in z.
        z = path if isinstance(path, dict) else np.load(path, allow_pickle=True)
        is_scale_100 = str(z.get("scale", "")) == "0-100"
        self.is_scale_100 = is_scale_100      # nhãn đã ở 0-100? -> quyết định label_scale khi ghi config
        self.feat = torch.tensor(z["feat"], dtype=torch.float32)
        self.gop_dim = self.feat.shape[-1]
        self.phn = torch.tensor(z["phn"].astype(np.int64))
        
        phn_sc = z["phone_label"] if "phone_label" in z else z["phn_score"]
        self.phone_label = torch.tensor(phn_sc, dtype=torch.float32)
        
        wacc_raw = torch.tensor(z["word_acc"], dtype=torch.float32)
        wacc = wacc_raw if is_scale_100 else wacc_raw / 5.0
        wid = torch.tensor(z["word_id"].astype(np.float32))
        
        if len(WORD_HEADS) == 1:
            self.word_label = torch.stack([wacc, wid], -1)                            # [N, L, 2]
        else:
            wtot_raw = torch.tensor(z["word_total"], dtype=torch.float32)
            wtot = wtot_raw if is_scale_100 else wtot_raw / 5.0
            self.word_label = torch.stack([wacc, wtot, wid], -1)                      # [N, L, 3]
            
        if "utt_label" in z:
            utt_raw = torch.tensor(z["utt_label"], dtype=torch.float32)
            self.utt_label = utt_raw if is_scale_100 else utt_raw / 5.0
        else:
            self.utt_label = torch.tensor(z["utt"][:, UTT_KEEP], dtype=torch.float32) / 5.0
        self.phone_list = list(z["phone_list"])
        self.use_occ = use_occ

        valid = self.phn >= 0                                                          # [N,50]
        vmask = valid.unsqueeze(-1).float()
        if feat_mean is None:
            fv = self.feat[valid]                                                      # [n_valid,41]
            feat_mean, feat_std = float(fv.mean()), max(float(fv.std()), 1e-6)
        self.feat_mean, self.feat_std = feat_mean, feat_std
        self.feat = ((self.feat - feat_mean) / feat_std) * vmask

        # occupancy as 42nd dim, normalized with its OWN scalar (different scale from LPP/LPR)
        if use_occ:
            if "occ" not in z:
                raise KeyError(f"{path} has no 'occ' array; re-extract with the updated gopt_data.py")
            occ = torch.tensor(z["occ"], dtype=torch.float32)                          # [N,50]
            if occ_mean is None:
                ov = occ[valid]
                occ_mean, occ_std = float(ov.mean()), float(ov.std())
            self.occ_mean, self.occ_std = occ_mean, occ_std
            occ = (((occ - occ_mean) / occ_std) * valid.float()).unsqueeze(-1)         # [N,50,1]
            self.feat = torch.cat([self.feat, occ], -1)                                # [N,50,42]
        else:
            self.occ_mean = self.occ_std = None

        # WavLM SSL feature (pooled per phone, PCA giảm chiều fit trên train) -> vào encoder
        self.use_wavlm = use_wavlm
        if use_wavlm:
            if "wavlm" not in z:
                raise KeyError(f"{path} chưa có 'wavlm'; chạy add_wavlm.py trước")
            wl = torch.tensor(z["wavlm"], dtype=torch.float32)                           # [N,50,1024]
            wv = wl[valid]                                                               # [n_valid,1024]
            if wavlm_pca is None:                                                        # fit trên train
                mu = wv.mean(0)
                U, Sv, Vh = torch.linalg.svd(wv - mu, full_matrices=False)
                comp = Vh[:wavlm_dim]                                                    # [dim,1024]
                wavlm_pca = (mu, comp)
            mu, comp = wavlm_pca
            self.wavlm_pca = (mu, comp)
            wl = (wl - mu) @ comp.T                                                      # [N,50,dim]
            wvp = wl[valid]
            if wavlm_norm is None:
                wavlm_norm = (wvp.mean(0), wvp.std(0).clamp_min(1e-6))
            wm, ws = wavlm_norm
            self.wavlm_norm = (wm, ws)
            self.wavlm_layer = int(z.get("wavlm_layer", 12))
            wl = ((wl - wm) / ws) * valid.unsqueeze(-1).float()
            self.feat = torch.cat([self.feat, wl], -1)                                   # [N,50,+dim]
        else:
            self.wavlm_pca = self.wavlm_norm = None
            self.wavlm_layer = None

        # prosody (3M): duration[1] + energy[7] = 8 dims, each normalized with its OWN scalar
        self.use_prosody = use_prosody
        if use_prosody:
            for k in ("dur", "eng"):
                if k not in z:
                    raise KeyError(f"{path} has no '{k}'; run add_prosody.py first")
            dur = torch.tensor(z["dur"], dtype=torch.float32).unsqueeze(-1)             # [N,50,1]
            eng = torch.tensor(z["eng"], dtype=torch.float32)                           # [N,50,7]
            pros = torch.cat([dur, eng], -1)                                            # [N,50,8]
            pv = pros[valid]                                                            # [n_valid,8]
            if pros_mean is None:
                pros_mean = pv.mean(0).tolist()
                pros_std = pv.std(0).clamp_min(1e-6).tolist()
            self.pros_mean, self.pros_std = pros_mean, pros_std
            pm = torch.tensor(pros_mean); ps = torch.tensor(pros_std)
            pros = ((pros - pm) / ps) * valid.unsqueeze(-1).float()
            self.feat = torch.cat([self.feat, pros], -1)                               # [N,50,+8]
        else:
            self.pros_mean = self.pros_std = None

    def __len__(self):
        return self.feat.size(0)

    def __getitem__(self, i):
        return {"feat": self.feat[i], "phn": self.phn[i], "phone_label": self.phone_label[i],
                "word_label": self.word_label[i], "utt_label": self.utt_label[i]}


def hf_split_to_npz_dict(hf_ds, use_wavlm=False, use_prosody=False):
    """Convert a HuggingFace dataset split into the dict-of-numpy layout GOPTDataset
    reads (same keys as the extract .npz). Only pulls heavy arrays actually needed."""
    cols = hf_ds.column_names
    nd = hf_ds.with_format("numpy")
    want = ["feat", "phn", "phone_label", "word_id", "word_acc", "utt_label",
            "phone_weight", "word_weight", "n_vendors", "utt_weight", "utt_nv",
            "msdd_type", "msdd_sub", "occ"]
    if use_prosody:
        want += ["dur", "eng"]
    if use_wavlm:
        want += ["wavlm"]
    out = {k: np.asarray(nd[k]) for k in want if k in cols}
    row0 = hf_ds[0]                                              # per-row-constant metadata
    out["phone_list"] = np.array(row0.get("phone_list") or [], dtype="U8")
    out["scale"] = np.array(str(row0.get("scale", "0-100")))
    if "wavlm_layer" in cols:
        out["wavlm_layer"] = np.int64(row0.get("wavlm_layer", 12))
    return out


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _agg_word(pred, target, word_id):
    """GOPT valid_word: average per-phone word preds/targets within each word_id."""
    wp, wt = [], []
    N, L = word_id.shape
    for i in range(N):
        prev, start = 0, 0
        for j in range(L):
            cur = int(word_id[i, j])
            if cur != prev:
                wp.append(pred[i, start:j].mean(0))
                wt.append(target[i, start:j].mean(0))
                if cur == -1:
                    break
                prev, start = cur, j
    return np.array(wp), np.array(wt).round(2)


def compute_metrics(ep):
    phone_p, word_p, utt_p = ep.predictions
    phone_l, word_l, utt_l = ep.label_ids
    mask = phone_l >= 0
    m = {"phone": pcc(phone_p, phone_l, mask)}
    # word: aggregate to word level by word_id (channel -1)
    wp, wt = _agg_word(word_p, word_l[..., :len(WORD_HEADS)], word_l[..., -1])
    for j, name in enumerate(WORD_HEADS):
        m[f"word_{name}"] = pcc(wp[:, j], wt[:, j], mask=(wt[:, j] >= 0))
    for j, name in enumerate(UTT_HEADS):
        m[f"utt_{name}"] = pcc(utt_p[:, j], utt_l[:, j], mask=(utt_l[:, j] >= 0))
    # near-constant on SO762 so its PCC is unstable/uninformative (see gopt_model.MEAN_HEADS).
    skip = {f"utt_{h}" for h in UTT_HEADS if h not in MEAN_HEADS}
    head_vals = {k: v for k, v in m.items() if k not in skip}          # 6 head: phone, word_*, utt_*
    m["mean"] = float(np.nanmean(list(head_vals.values())))
    # mean nghiêng phone: phone nhân PHONE_W, còn lại 1 (nan-safe) -> dùng cho --best-metric phone_heavy
    w = {k: (PHONE_W if k == "phone" else 1.0) for k in head_vals}
    num = float(np.nansum([w[k] * head_vals[k] for k in head_vals]))
    den = float(np.nansum([w[k] * (0.0 if np.isnan(head_vals[k]) else 1.0) for k in head_vals]))
    m["phone_heavy"] = num / den if den > 0 else float("nan")
    return m


def flatten_dict(d):
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[sub_k.replace("-", "_")] = sub_v
        else:
            flat[k.replace("-", "_")] = v
    return flat


def main():
    pre_p = argparse.ArgumentParser(add_help=False)
    pre_p.add_argument("--config", default=None)
    pre_args, _ = pre_p.parse_known_args()

    cfg = {}
    if pre_args.config and os.path.exists(pre_args.config):
        from vh_gopt.config import load_config_file
        cfg = flatten_dict(load_config_file(pre_args.config))

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=pre_args.config, help="Path to YAML/JSON experiment config")
    ap.add_argument("--train", default=cfg.get("train", "data/gopt_vh_scripted_gold/train.npz"))
    ap.add_argument("--test", default=cfg.get("test", "data/gopt_vh_scripted_gold/test_unseen_speakers.npz"))
    ap.add_argument("--val", default=cfg.get("val", None),
                    help="val split (npz path or split name if --hf-dataset). Best checkpoint selected on this; falls back to test if unset.")
    ap.add_argument("--test2", default=cfg.get("test2", None),
                    help="second test split (e.g. test_unseen_prompts) reported at the end.")
    ap.add_argument("--hf-dataset", default=cfg.get("hf_dataset", None),
                    help="HF dataset repo (e.g. tiennguyenbnbk/gopt-vh-gold-features); when set, --train/--val/--test/--test2 are treated as SPLIT NAMES.")
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 80))
    ap.add_argument("--bs", type=int, default=cfg.get("bs", 25))
    ap.add_argument("--lr", type=float, default=cfg.get("lr", 1e-3))
    ap.add_argument("--wd", type=float, default=cfg.get("wd", 5e-7))
    ap.add_argument("--embed-dim", type=int, default=cfg.get("embed_dim", 24))
    ap.add_argument("--heads", type=int, default=cfg.get("heads", 1))
    ap.add_argument("--depth", type=int, default=cfg.get("depth", 3))
    ap.add_argument("--model", choices=["gopt", "hia"], default=cfg.get("model", "gopt"),
                    help="hia = Residual Hierarchical Interactive Attention (arXiv:2601.01745)")
    ap.add_argument("--arch", choices=["base", "mlp", "concat", "film"], default=cfg.get("arch", "base"),
                    help="(gopt only) phone-conditioning / input projection variant")
    ap.add_argument("--phono", action="store_true", default=cfg.get("phono", False))
    ap.add_argument("--think", type=int, default=cfg.get("think", 0))
    ap.add_argument("--attn-pool", action="store_true", default=cfg.get("attn_pool", False))
    ap.add_argument("--dropout", type=float, default=cfg.get("dropout", 0.20))
    ap.add_argument("--sched", choices=["gopt", "cosine"], default=cfg.get("sched", "cosine"))
    ap.add_argument("--balanced", action="store_true", default=cfg.get("balanced", False))
    ap.add_argument("--balance-heads", choices=["all", "phn"], default=cfg.get("balance_heads", "phn"))
    ap.add_argument("--bins", type=int, default=cfg.get("bins", 10))
    ap.add_argument("--beta", type=float, default=cfg.get("beta", 0.999))
    ap.add_argument("--wcap", type=float, default=cfg.get("wcap", 5.0))
    ap.add_argument("--gop-map", default=cfg.get("gop_map", "koel"), choices=["arpa", "koel"])
    ap.add_argument("--acoustic-model", default=cfg.get("acoustic_model", "KoelLabs/xlsr-english-01"))
    ap.add_argument("--use-occ", action="store_true", default=cfg.get("use_occ", False))
    ap.add_argument("--use-prosody", action="store_true", default=cfg.get("use_prosody", False))
    ap.add_argument("--utt-prosody", action=argparse.BooleanOptionalAction, default=cfg.get("utt_prosody", True))
    ap.add_argument("--use-wavlm", action="store_true", default=cfg.get("use_wavlm", True))
    ap.add_argument("--wavlm-dim", type=int, default=cfg.get("wavlm_dim", 32))
    ap.add_argument("--wavlm-fuse", choices=["stack", "phone", "utt"], default=cfg.get("wavlm_fuse", "stack"))
    ap.add_argument("--best-metric", choices=["mean", "phone", "phone_heavy"], default=cfg.get("best_metric", "mean"))
    ap.add_argument("--phone-weight", type=float, default=cfg.get("phone_weight", 2.0))
    ap.add_argument("--noise", type=float, default=cfg.get("noise", 0.10))
    ap.add_argument("--w-phn", type=float, default=cfg.get("w_phn", 1.0))
    ap.add_argument("--w-word", type=float, default=cfg.get("w_word", 1.0))
    ap.add_argument("--w-utt", type=float, default=cfg.get("w_utt", 1.0))
    ap.add_argument("--out", default=cfg.get("out", "ckpt/stage2_baseline_wavlm32"))
    ap.add_argument("--early-stop-patience", type=int, default=cfg.get("early_stop_patience", 0),
                    help="stop if val metric_for_best_model doesn't improve for N evals (0 = off).")
    ap.add_argument("--early-stop-threshold", type=float, default=cfg.get("early_stop_threshold", 0.0),
                    help="min improvement to reset patience.")
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    ap.add_argument("--wandb-project", default=cfg.get("wandb_project", "gop-ctc-gopt"))
    ap.add_argument("--wandb-run", default=cfg.get("wandb_run", None))
    ap.add_argument("--no-wandb", action="store_true", default=cfg.get("no_wandb", False))
    ap.add_argument("--push-model", action="store_true", default=cfg.get("push_model", False))
    ap.add_argument("--hf-repo", default=cfg.get("hf_repo", None))
    args = ap.parse_args()
    global PHONE_W
    PHONE_W = args.phone_weight

    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(args.seed)

    use_wandb = not args.no_wandb
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    assert not (args.use_prosody and args.utt_prosody), "pick one prosody mode"
    load_pros = args.use_prosody or args.utt_prosody

    # Resolve data sources: either HF dataset splits (built into dict-of-arrays) or npz paths.
    if args.hf_dataset:
        from datasets import load_dataset
        def _sp(v, default):
            return default if (not v or v.endswith(".npz") or "/" in v) else v
        train_split = _sp(args.train, "train")
        val_split = _sp(args.val, "val")
        test_split = _sp(args.test, "test_unseen_speakers")
        test2_split = _sp(args.test2, "test_unseen_prompts")
        print(f"Nạp HF dataset: {args.hf_dataset}  splits: train={train_split} val={val_split} "
              f"test={test_split} test2={test2_split}")
        dd = load_dataset(args.hf_dataset)
        def _src(split):
            return hf_split_to_npz_dict(dd[split], use_wavlm=args.use_wavlm, use_prosody=load_pros)
        train_src, val_src = _src(train_split), (_src(val_split) if val_split in dd else None)
        test_src = _src(test_split)
        test2_src = _src(test2_split) if test2_split in dd else None
    else:
        train_src, val_src = args.train, (args.val or None)
        test_src, test2_src = args.test, (args.test2 or None)

    tr = GOPTDataset(train_src, use_occ=args.use_occ, use_prosody=load_pros,
                     use_wavlm=args.use_wavlm, wavlm_dim=args.wavlm_dim)
    def _mk(src):                                              # apply TRAIN stats to eval splits
        return GOPTDataset(src, feat_mean=tr.feat_mean, feat_std=tr.feat_std,
                           use_occ=args.use_occ, occ_mean=tr.occ_mean, occ_std=tr.occ_std,
                           use_prosody=load_pros, pros_mean=tr.pros_mean, pros_std=tr.pros_std,
                           use_wavlm=args.use_wavlm, wavlm_dim=args.wavlm_dim,
                           wavlm_pca=tr.wavlm_pca, wavlm_norm=tr.wavlm_norm)
    te = _mk(test_src)
    va = _mk(val_src) if val_src is not None else te          # select best on val (fallback: test)
    te2 = _mk(test2_src) if test2_src is not None else None
    if val_src is None:
        print("[WARN] no val split -> selecting best checkpoint on TEST (leakage). Pass --val to fix.")
    gop_dim = tr.gop_dim                                            # 41 (./model) hoặc 80 (KoelLabs)
    wavlm_dim = args.wavlm_dim if args.use_wavlm else 0
    enc_dim = gop_dim + (1 if args.use_occ else 0) + wavlm_dim      # encoder feature width
    # use_prosody: prosody feeds the encoder (all heads). utt_prosody: encoder stays enc_dim,
    # prosody (8d) rides along x's tail and is peeled off for the utterance branch only.
    input_dim = enc_dim + (8 if args.use_prosody else 0)
    prosody_dim = 8 if args.utt_prosody else 0
    print(f"train={len(tr)}  test={len(te)}  enc_dim={input_dim}  utt_prosody={args.utt_prosody}  "
          f"sched={args.sched}  balanced={args.balanced}  feat_norm(mean={tr.feat_mean:.3f}, std={tr.feat_std:.3f})"
          + (f"  occ_norm(mean={tr.occ_mean:.3f}, std={tr.occ_std:.3f})" if args.use_occ else "")
          + ("  +prosody(8d,enc)" if args.use_prosody else "")
          + ("  +prosody(8d,utt)" if args.utt_prosody else ""))

    bw = {}
    if args.balanced:
        hi100 = 100.0 if tr.is_scale_100 else 2.0                   # dải điểm thật: 0-100 (gold) hay 0-2 (SO762)
        vphn = tr.phn.numpy() >= 0
        edges, wphn = effective_number_weights(tr.phone_label.numpy()[vphn], args.bins, args.beta, hi=hi100, wcap=args.wcap)
        if args.balance_heads == "all":
            _, wutt = effective_number_weights(tr.utt_label.numpy().reshape(-1), args.bins, args.beta, hi=hi100, wcap=args.wcap)
        else:
            wutt = np.ones_like(wphn)                               # plain mean for utt
        bw = dict(balanced=True, bw_edges=edges, bw_phn=wphn, bw_utt=wutt)
        print(f"balance heads={args.balance_heads}  weights: phn min/max={wphn.min():.2f}/{wphn.max():.2f}  "
              f"utt min/max={wutt.min():.2f}/{wutt.max():.2f}")

    if args.model == "hia":
        model = HIAForScoring(input_dim=input_dim, embed_dim=args.embed_dim, num_heads=args.heads,
                              depth=args.depth, dropout=args.dropout, noise=args.noise,
                              w_phn=args.w_phn, w_word=args.w_word, w_utt=args.w_utt)
        print(f"model=HIA embed={args.embed_dim} heads={args.heads} depth={args.depth}  "
              f"params={sum(p.numel() for p in model.parameters())}")
    else:
        jcapt = {}
        if args.phono:
            from phono import phono_buffer
            jcapt = dict(use_phono=True, phono_matrix=phono_buffer(tr.phone_list, 40))
        model = GOPTForScoring(input_dim=input_dim, embed_dim=args.embed_dim, num_heads=args.heads,
                               depth=args.depth, dropout=args.dropout, arch=args.arch, noise=args.noise,
                               n_think=args.think, attn_pool=args.attn_pool,
                               utt_prosody=args.utt_prosody, prosody_dim=prosody_dim,
                               wavlm_dim=wavlm_dim, wavlm_fuse=args.wavlm_fuse,
                               w_phn=args.w_phn, w_word=args.w_word, w_utt=args.w_utt, **jcapt, **bw)
        print(f"model=GOPT arch={args.arch} phono={args.phono} think={args.think} "
              f"attn_pool={args.attn_pool} utt_prosody={args.utt_prosody}  "
              f"params={sum(p.numel() for p in model.parameters())}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=256,
        learning_rate=args.lr,
        weight_decay=args.wd,
        adam_beta1=0.95, adam_beta2=0.999,          # GOPT Adam betas
        lr_scheduler_type=("cosine" if args.sched == "cosine" else "constant"),
        warmup_steps=100,                            # GOPT warms first 100 steps
        max_grad_norm=5.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.best_metric}",
        greater_is_better=True,
        report_to=(["wandb"] if use_wandb else ["none"]),
        run_name=args.wandb_run,
        remove_unused_columns=False,
        label_names=LABEL_NAMES,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    optimizers = (None, None)
    if args.sched == "gopt":
        import math
        spe = math.ceil(len(tr) / args.bs)                         # steps per epoch
        milestones = [e * spe for e in range(20, args.epochs, 5)]  # halve every 5 ep after ep20
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd,
                               betas=(0.95, 0.999))
        def lr_lambda(step):
            warm = min((step + 1) / 100.0, 1.0)                    # 100-step warmup
            return warm * (0.5 ** sum(step >= ms for ms in milestones))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        optimizers = (opt, sched)

    callbacks = []
    if args.early_stop_patience and args.early_stop_patience > 0:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience,
                                               early_stopping_threshold=args.early_stop_threshold))
        print(f"early stopping: patience={args.early_stop_patience} threshold={args.early_stop_threshold} "
              f"on val eval_{args.best_metric}")
    trainer = Trainer(model=model, args=targs, train_dataset=tr, eval_dataset=va,
                      data_collator=collate, compute_metrics=compute_metrics,
                      optimizers=optimizers, callbacks=callbacks)
    trainer.train()

    drop = ("eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second")
    def _eval(ds, prefix):
        m = trainer.evaluate(ds, metric_key_prefix=prefix)
        return {k.replace(f"{prefix}_", ""): v for k, v in m.items()
                if k.startswith(f"{prefix}_") and k not in
                (f"{prefix}_loss", f"{prefix}_runtime", f"{prefix}_samples_per_second", f"{prefix}_steps_per_second")}

    # best model (selected on VAL) is loaded; report on both held-out test splits
    val_pcc = _eval(va, "val")
    best = _eval(te, "test")                       # primary test = test_unseen_speakers
    all_test = {"test_unseen_speakers": best}
    print("\nval PCC:", json.dumps({k: round(v, 4) for k, v in val_pcc.items()}, ensure_ascii=False))
    print("final test (unseen_speakers) PCC:", json.dumps({k: round(v, 4) for k, v in best.items()}, ensure_ascii=False))
    if te2 is not None:
        test2 = _eval(te2, "test2")
        all_test["test_unseen_prompts"] = test2
        print("final test (unseen_prompts) PCC:", json.dumps({k: round(v, 4) for k, v in test2.items()}, ensure_ascii=False))

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    _write_config(args, best, tr, val_pcc=val_pcc, all_test=all_test)
    if use_wandb:
        import wandb
        if wandb.run is not None:
            wandb.summary["val_mean_pcc"] = val_pcc.get("mean")
            wandb.summary["test_mean_pcc"] = best.get("mean")
            for split, d in all_test.items():
                for k, v in d.items():
                    wandb.summary[f"{split}/{k}"] = v
            wandb.finish()

    if args.push_model:
        if not args.hf_repo:
            print("[push skipped: pass --hf-repo]")
        else:
            _push(args.out, args.hf_repo, best)


def _write_config(args, best, tr, val_pcc=None, all_test=None):
    wavlm_dim = args.wavlm_dim if args.use_wavlm else 0
    if args.use_wavlm:                                    # lưu PCA + norm cho inference
        mu, comp = tr.wavlm_pca; wm, ws = tr.wavlm_norm
        np.savez(os.path.join(args.out, "wavlm_pca.npz"),
                 mean=mu.numpy(), comp=comp.numpy(), norm_mean=wm.numpy(), norm_std=ws.numpy())
    cfg = {"arch": "GOPT", "model": args.model, "arch_variant": args.arch,
           "gop_dim": tr.gop_dim, "gop_map": args.gop_map, "acoustic_model": args.acoustic_model,
           "use_wavlm": args.use_wavlm, "wavlm_dim": wavlm_dim, "wavlm_fuse": args.wavlm_fuse,
           "wavlm_model": "microsoft/wavlm-large", "wavlm_layer": tr.wavlm_layer,
           "input_dim": tr.gop_dim + (1 if args.use_occ else 0) + wavlm_dim + (8 if args.use_prosody else 0),
           "embed_dim": args.embed_dim,
           "num_heads": args.heads, "depth": args.depth, "dropout": args.dropout,
           "max_len": 150, "n_phn_cls": 40, "use_occ": args.use_occ,   # GOPT dùng default max_len=150 (pos-embed); trước ghi nhầm 50
           "use_prosody": args.use_prosody, "utt_prosody": args.utt_prosody,
           "prosody_dim": 8 if args.utt_prosody else 0,
           "use_phono": args.phono, "n_think": args.think, "attn_pool": args.attn_pool,
           "utt_heads": list(UTT_HEADS), "word_heads": list(WORD_HEADS),
           "phone_list": tr.phone_list,
           "feat_norm": {"mean": tr.feat_mean, "std": tr.feat_std},
           # nhãn 0-100 -> model xuất thẳng 0-100, KHÔNG nhân (to_100=1.0). Nhãn 0-2 (SO762) -> x50.
           # Trước đây ghi cứng to_100=50 -> inference nhân đôi -> điểm clip bão hoà 100. (bug đã sửa)
           "label_scale": ({"phone": 1.0, "word": 1.0, "utt": 1.0, "to_100": 1.0}
                           if tr.is_scale_100 else
                           {"phone": 1.0, "word": 5.0, "utt": 5.0, "to_100": 50.0}),
           "hf_dataset": args.hf_dataset,
           "train_cfg": {"sched": args.sched, "balanced": args.balanced,
                         "epochs": args.epochs, "bins": args.bins, "beta": args.beta},
           "selection": "val" if args.val else "test",
           "val_pcc": val_pcc, "test_pcc": best,
           "all_test_pcc": all_test or {"test_unseen_speakers": best}}
    if args.use_occ:
        cfg["occ_norm"] = {"mean": tr.occ_mean, "std": tr.occ_std}
    if args.use_prosody or args.utt_prosody:
        cfg["pros_norm"] = {"mean": tr.pros_mean, "std": tr.pros_std}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    from vh_gopt.config import save_config
    save_config(vars(args), os.path.join(args.out, "config.yaml"))

def _push(ckpt_dir, repo, best):
    from huggingface_hub import HfApi
    api = HfApi()
    if "/" not in repo:
        repo = f"{api.whoami()['name']}/{repo}"
    cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    readme = f"""---
tags: [pronunciation-assessment, gop, gopt, speechocean762]
library_name: pytorch
---

# GOPT on CTC-GOP features

Transformer pronunciation scorer (GOPT) trained with `transformers.Trainer` on
41-d CTC-GOP features (wav2vec2 phoneme CTC, taslpro26 GOP-SF-Norm) from
Speechocean762. Data handling & loss follow YuanGongND/gopt.

- input: `x [B,50,41]` GOP feature (global-scalar normalized) + `phn [B,50]` phone id
- heads: phone accuracy; word (accuracy, total); utterance (accuracy, completeness, fluency, total)
- labels on 0-2 scale (word/utt /5); x50 -> 0-100.
- word PCC evaluated at word level (per-word averaging by word_id).

## Test PCC
```json
{json.dumps(cfg['test_pcc'], indent=2)}
```
Reload: `gopt_train.GOPTForScoring(**cfg)` + `safetensors.torch.load_model`; normalize
input with `config.feat_norm` before inference.
"""
    open(os.path.join(ckpt_dir, "README.md"), "w").write(readme)
    api.create_repo(repo, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(folder_path=ckpt_dir, repo_id=repo, repo_type="model",
                      ignore_patterns=["checkpoint-*/**", "runs/**"])
    # also push the model source ("mã tương ứng") so the checkpoint is self-reloadable
    import vh_gopt.training as _t
    src_dir = os.path.dirname(_t.__file__)
    for fn in ("gopt_model.py", "gopt_hia.py", "gopt_train.py"):
        fp = os.path.join(src_dir, fn)
        if os.path.exists(fp):
            api.upload_file(path_or_fileobj=fp, path_in_repo=f"code/{fn}",
                            repo_id=repo, repo_type="model")
    print(f"pushed model + code -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
