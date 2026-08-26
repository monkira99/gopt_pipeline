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
        idx = torch.bucketize(label.clamp(0, 2.0), self.bw_edges)   # bin index
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
        z = np.load(path, allow_pickle=True)
        is_scale_100 = str(z.get("scale", "")) == "0-100"
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
        m[f"word_{name}"] = pcc(wp[:, j], wt[:, j])
    for j, name in enumerate(UTT_HEADS):
        m[f"utt_{name}"] = pcc(utt_p[:, j], utt_l[:, j])
    # completeness kept as an auxiliary training head but excluded from `mean`: its label is
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/gopt/train.npz")
    ap.add_argument("--test", default="data/gopt/test.npz")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-7)          # GOPT weight_decay
    ap.add_argument("--embed-dim", type=int, default=24)       # GOPT paper best
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--model", choices=["gopt", "hia"], default="gopt",
                    help="hia = Residual Hierarchical Interactive Attention (arXiv:2601.01745)")
    ap.add_argument("--arch", choices=["base", "mlp", "concat", "film"], default="base",
                    help="(gopt only) phone-conditioning / input projection variant")
    ap.add_argument("--phono", action="store_true", help="JCAPT phonological attr features")
    ap.add_argument("--think", type=int, default=0, help="JCAPT think tokens (0=off)")
    ap.add_argument("--attn-pool", action="store_true", help="JCAPT per-aspect attention pooling for utt")
    ap.add_argument("--dropout", type=float, default=0.1)     # 0.1 beats GOPT's 0 on our features
    ap.add_argument("--sched", choices=["gopt", "cosine"], default="cosine",
                    help="cosine wins here; gopt = MultiStepLR halve every 5 ep after ep20 (paper)")
    ap.add_argument("--balanced", action="store_true", help="score-balanced loss for label skew")
    ap.add_argument("--balance-heads", choices=["all", "phn"], default="phn",
                    help="phn = balance phone/word only (utt stays plain mean)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.999)
    ap.add_argument("--wcap", type=float, default=5.0)
    ap.add_argument("--gop-map", default="arpa", choices=["arpa", "koel"],
                    help="ghi vào config để inference biết cách map âm: arpa (./model) | koel (Path-A GOP-79d)")
    ap.add_argument("--acoustic-model", default="./model",
                    help="đường dẫn acoustic model dùng khi trích GOP (ghi vào config cho inference)")
    ap.add_argument("--use-occ", action="store_true", help="add occupancy as 42nd feature dim")
    ap.add_argument("--use-prosody", action="store_true", help="3M duration+energy (8d) into encoder (all heads)")
    ap.add_argument("--utt-prosody", action=argparse.BooleanOptionalAction, default=True,
                    help="3M duration+energy (8d) into utterance branch ONLY (new baseline, default on; "
                         "--no-utt-prosody to disable). Needs dur/eng in npz (run add_prosody.py).")
    ap.add_argument("--use-wavlm", action="store_true",
                    help="fuse WavLM SSL feature (pooled/phone, PCA) vào encoder. Cần add_wavlm.py trước")
    ap.add_argument("--wavlm-dim", type=int, default=128, help="chiều PCA giảm cho WavLM (mặc định 128)")
    ap.add_argument("--wavlm-fuse", choices=["stack", "phone", "utt"], default="stack",
                    help="cách fuse WavLM: stack=nhồi chung core (cũ); phone=projection riêng cộng vào token phone; "
                         "utt=pool projection riêng CHỈ vào nhánh utt (phone thuần GOP)")
    ap.add_argument("--best-metric", choices=["mean", "phone", "phone_heavy"], default="mean",
                    help="tiêu chí chọn best checkpoint: mean=trung bình 6 head (cũ); phone=chỉ phone; "
                         "phone_heavy=mean nghiêng phone (phone nhân --phone-weight)")
    ap.add_argument("--phone-weight", type=float, default=2.0,
                    help="trọng số phone trong metric phone_heavy (mặc định 2.0)")
    ap.add_argument("--noise", type=float, default=0.0)        # GOPT input aug
    ap.add_argument("--w-phn", type=float, default=1.0)
    ap.add_argument("--w-word", type=float, default=1.0)
    ap.add_argument("--w-utt", type=float, default=1.0)
    ap.add_argument("--out", default="ckpt/gopt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="gop-ctc-gopt")
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--push-model", action="store_true")
    ap.add_argument("--hf-repo", default=None, help="e.g. gopt-ctc-gop (namespace auto)")
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
    tr = GOPTDataset(args.train, use_occ=args.use_occ, use_prosody=load_pros,
                     use_wavlm=args.use_wavlm, wavlm_dim=args.wavlm_dim)
    te = GOPTDataset(args.test, feat_mean=tr.feat_mean, feat_std=tr.feat_std,
                     use_occ=args.use_occ, occ_mean=tr.occ_mean, occ_std=tr.occ_std,
                     use_prosody=load_pros, pros_mean=tr.pros_mean, pros_std=tr.pros_std,
                     use_wavlm=args.use_wavlm, wavlm_dim=args.wavlm_dim,
                     wavlm_pca=tr.wavlm_pca, wavlm_norm=tr.wavlm_norm)  # train stats
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
        vphn = tr.phn.numpy() >= 0
        edges, wphn = effective_number_weights(tr.phone_label.numpy()[vphn], args.bins, args.beta, wcap=args.wcap)
        if args.balance_heads == "all":
            _, wutt = effective_number_weights(tr.utt_label.numpy().reshape(-1), args.bins, args.beta, wcap=args.wcap)
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

    trainer = Trainer(model=model, args=targs, train_dataset=tr, eval_dataset=te,
                      data_collator=collate, compute_metrics=compute_metrics, optimizers=optimizers)
    trainer.train()

    metrics = trainer.evaluate()
    drop = ("eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second")
    best = {k.replace("eval_", ""): v for k, v in metrics.items()
            if k.startswith("eval_") and k not in drop}
    print("\nfinal test PCC:", json.dumps({k: round(v, 4) for k, v in best.items()}, ensure_ascii=False))

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    _write_config(args, best, tr)
    if use_wandb:
        import wandb
        if wandb.run is not None:
            wandb.summary["best_mean_pcc"] = best.get("mean")
            wandb.finish()

    if args.push_model:
        if not args.hf_repo:
            print("[push skipped: pass --hf-repo]")
        else:
            _push(args.out, args.hf_repo, best)


def _write_config(args, best, tr):
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
           "max_len": 50, "n_phn_cls": 40, "use_occ": args.use_occ,
           "use_prosody": args.use_prosody, "utt_prosody": args.utt_prosody,
           "prosody_dim": 8 if args.utt_prosody else 0,
           "use_phono": args.phono, "n_think": args.think, "attn_pool": args.attn_pool,
           "utt_heads": list(UTT_HEADS), "word_heads": list(WORD_HEADS),
           "phone_list": tr.phone_list,
           "feat_norm": {"mean": tr.feat_mean, "std": tr.feat_std},
           "label_scale": {"phone": 1.0, "word": 5.0, "utt": 5.0, "to_100": 50.0},
           "train_cfg": {"sched": args.sched, "balanced": args.balanced,
                         "epochs": args.epochs, "bins": args.bins, "beta": args.beta},
           "test_pcc": best}
    if args.use_occ:
        cfg["occ_norm"] = {"mean": tr.occ_mean, "std": tr.occ_std}
    if args.use_prosody or args.utt_prosody:
        cfg["pros_norm"] = {"mean": tr.pros_mean, "std": tr.pros_std}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)


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
    print(f"pushed model -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
