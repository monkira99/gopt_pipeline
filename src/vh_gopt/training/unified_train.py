#!/usr/bin/env python3
"""Train mô hình HỢP NHẤT (GOPT scorer + MSDD detector, 1 trunk) — xem unified_model.py.

Kế thừa hạ tầng stage-2 (gopt_train): cùng GOPTDataset/normalize/WavLM-PCA/HF-Trainer,
nên tự có: chọn best trên VAL (không leak), --seed đa hạt, trọng số loss theo config.
Thêm so với gopt_train:
  - dataset use_msdd=True (nạp msdd_type/msdd_sub, thêm vào label_names).
  - model GOPTUnifiedForScoring; class-weight detection (sqrt-inverse-freq) tính trên train.
  - compute_metrics: PCC (phone/word/utt) + MSDD (det F1/FAR/FRR, diag acc) trong một lần eval.
  - best_metric: 'unified' = 0.5*mean_PCC + 0.5*det_f1 (mặc định — giữ PCC, không bỏ rơi MSDD).

Usage:
  python -m vh_gopt.training.unified_train --config configs/stage2/unified.yaml
"""
import argparse, json, os, math
import numpy as np
import torch

from vh_gopt.training.gopt_model import UTT_HEADS, WORD_HEADS, MEAN_HEADS
from vh_gopt.training.gopt_train import (
    GOPTDataset, hf_split_to_npz_dict, collate, pcc, _agg_word, flatten_dict,
)
from vh_gopt.training.unified_model import GOPTUnifiedForScoring, PHONE_NUM

LABEL_NAMES = ["phone_label", "word_label", "utt_label", "msdd_type", "msdd_sub"]
PHONE_W = 2.0


def _msdd_metrics(det_p, det_t, diag_p, diag_t):
    """Khớp convention stage1_mdd_train: FAR=chấm oan (fp/(fp+tn)), FRR=bỏ sót (fn/(fn+tp))."""
    v = det_t >= 0
    yt, yp = det_t[v], det_p[v]
    te, pe = (yt > 0).astype(int), (yp > 0).astype(int)
    tp = int(((pe == 1) & (te == 1)).sum()); fp = int(((pe == 1) & (te == 0)).sum())
    tn = int(((pe == 0) & (te == 0)).sum()); fn = int(((pe == 0) & (te == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    far = fp / max(fp + tn, 1); frr = fn / max(fn + tp, 1)
    acc3 = float((yt == yp).mean()) if len(yt) else 0.0
    delr = int(((yt == 2) & (yp == 2)).sum()) / max(int((yt == 2).sum()), 1)
    subr = int(((yt == 1) & (yp == 1)).sum()) / max(int((yt == 1).sum()), 1)
    vd = diag_t >= 0
    dex = float((diag_p[vd] == diag_t[vd]).mean()) if int(vd.sum()) else 0.0
    return dict(det_f1=f1, det_precision=prec, det_recall=rec, det_far=far, det_frr=frr,
                det_acc3=acc3, del_recall=delr, sub_recall=subr, diag_acc=dex)


def _coherence(phone_p, det_p, det_t, score_scale):
    """Đo trực tiếp lo ngại 'sai âm mà điểm cao': trên phone MÔ HÌNH gắn cờ lỗi (pred>0),
    %điểm >= 60% thang, và corr(score, pred_error) (nên ÂM mạnh)."""
    v = det_t >= 0
    sc = phone_p[v]; err = (det_p[v] > 0).astype(float)
    pf = err > 0
    hi = float((sc[pf] >= 0.6 * score_scale).mean()) if pf.sum() else 0.0
    coh = pcc(sc, err) if v.sum() > 2 else float("nan")
    return dict(coh_score_err=coh, pct_flag_hi=hi)


def make_compute_metrics(score_scale):
    def compute_metrics(ep):
        phone_p, word_p, utt_p, det_l, diag_l = ep.predictions
        phone_l, word_l, utt_l, mtype_l, msub_l = ep.label_ids
        mask = phone_l >= 0
        m = {"phone": pcc(phone_p, phone_l, mask)}
        wp, wt = _agg_word(word_p, word_l[..., :len(WORD_HEADS)], word_l[..., -1])
        for j, name in enumerate(WORD_HEADS):
            m[f"word_{name}"] = pcc(wp[:, j], wt[:, j], mask=(wt[:, j] >= 0))
        for j, name in enumerate(UTT_HEADS):
            m[f"utt_{name}"] = pcc(utt_p[:, j], utt_l[:, j], mask=(utt_l[:, j] >= 0))
        skip = {f"utt_{h}" for h in UTT_HEADS if h not in MEAN_HEADS}
        head_vals = {k: v for k, v in m.items() if k not in skip}
        m["mean"] = float(np.nanmean(list(head_vals.values())))
        # MSDD
        det_p = det_l.argmax(-1); diag_p = diag_l.argmax(-1)
        m.update(_msdd_metrics(det_p, mtype_l, diag_p, msub_l))
        m.update(_coherence(phone_p, det_p, mtype_l, score_scale))
        # tiêu chí gộp: giữ PCC + không bỏ rơi MSDD
        m["unified"] = 0.5 * m["mean"] + 0.5 * m["det_f1"]
        return m
    return compute_metrics


def main():
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = {}
    if pre_args.config and os.path.exists(pre_args.config):
        from vh_gopt.config import load_config_file
        cfg = flatten_dict(load_config_file(pre_args.config))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=pre_args.config)
    ap.add_argument("--train", default=cfg.get("train", "train"))
    ap.add_argument("--val", default=cfg.get("val", "val"))
    ap.add_argument("--test", default=cfg.get("test", "test_unseen_speakers"))
    ap.add_argument("--test2", default=cfg.get("test2", "test_unseen_prompts"))
    ap.add_argument("--hf-dataset", default=cfg.get("hf_dataset", "tiennguyenbnbk/gopt-vh-gold-features-v2"))
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 80))
    ap.add_argument("--bs", type=int, default=cfg.get("bs", 25))
    ap.add_argument("--lr", type=float, default=cfg.get("lr", 1e-3))
    ap.add_argument("--wd", type=float, default=cfg.get("wd", 1e-4))
    ap.add_argument("--embed-dim", type=int, default=cfg.get("embed_dim", 48))
    ap.add_argument("--heads", type=int, default=cfg.get("heads", 4))
    ap.add_argument("--depth", type=int, default=cfg.get("depth", 4))
    ap.add_argument("--arch", choices=["base", "mlp", "concat", "film"], default=cfg.get("arch", "base"))
    ap.add_argument("--phono", action="store_true", default=cfg.get("phono", False))
    ap.add_argument("--think", type=int, default=cfg.get("think", 0))
    ap.add_argument("--attn-pool", action="store_true", default=cfg.get("attn_pool", False))
    ap.add_argument("--dropout", type=float, default=cfg.get("dropout", 0.20))
    ap.add_argument("--sched", choices=["gopt", "cosine"], default=cfg.get("sched", "cosine"))
    ap.add_argument("--use-occ", action="store_true", default=cfg.get("use_occ", False))
    ap.add_argument("--use-prosody", action="store_true", default=cfg.get("use_prosody", False))
    ap.add_argument("--utt-prosody", action=argparse.BooleanOptionalAction, default=cfg.get("utt_prosody", True))
    ap.add_argument("--use-wavlm", action="store_true", default=cfg.get("use_wavlm", True))
    ap.add_argument("--wavlm-dim", type=int, default=cfg.get("wavlm_dim", 32))
    ap.add_argument("--wavlm-fuse", choices=["stack", "phone", "utt"], default=cfg.get("wavlm_fuse", "stack"))
    ap.add_argument("--wavlm-proj", choices=["pca", "linear"], default=cfg.get("wavlm_proj", "pca"),
                    help="pca=SVD-32 cố định (mặc định); linear=giữ WavLM 1024 thô, in_proj học projection end-to-end.")
    ap.add_argument("--feat-norm", choices=["scalar", "perdim"], default=cfg.get("feat_norm", "scalar"),
                    help="scalar=1 mean/std cho cả GOP (production); perdim=mỗi chiều GOP một mean/std.")
    ap.add_argument("--noise", type=float, default=cfg.get("noise", 0.10))
    ap.add_argument("--w-phn", type=float, default=cfg.get("w_phn", 1.0))
    ap.add_argument("--w-word", type=float, default=cfg.get("w_word", 1.0))
    ap.add_argument("--w-utt", type=float, default=cfg.get("w_utt", 1.0))
    ap.add_argument("--w-det", type=float, default=cfg.get("w_det", 1.0))
    ap.add_argument("--w-diag", type=float, default=cfg.get("w_diag", 1.0))
    ap.add_argument("--w-cons", type=float, default=cfg.get("w_cons", 0.0),
                    help="consistency weight; đã chứng minh thừa -> mặc định 0.")
    ap.add_argument("--det-wpow", type=float, default=cfg.get("det_wpow", 0.5),
                    help="số mũ trọng số lớp detection (0.5=sqrt-inverse-freq, 0=đều).")
    ap.add_argument("--no-word-head", action="store_true", default=cfg.get("no_word_head", True))
    ap.add_argument("--best-metric", default=cfg.get("best_metric", "unified"),
                    help="unified|mean|phone|det_f1|diag_acc ...")
    ap.add_argument("--early-stop-patience", type=int, default=cfg.get("early_stop_patience", 12))
    ap.add_argument("--early-stop-threshold", type=float, default=cfg.get("early_stop_threshold", 0.0))
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    ap.add_argument("--out", default=cfg.get("out", "ckpt/unified"))
    ap.add_argument("--wandb-project", default=cfg.get("wandb_project", "gopt-vh-experiments"))
    ap.add_argument("--wandb-run", default=cfg.get("wandb_run", None))
    ap.add_argument("--no-wandb", action="store_true", default=cfg.get("no_wandb", True))
    ap.add_argument("--push-model", action=argparse.BooleanOptionalAction, default=cfg.get("push_model", False))
    ap.add_argument("--hf-repo", default=cfg.get("hf_repo", None))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()

    if args.no_word_head:
        args.w_word = 0.0
        print("[no_word_head] word = analytic mean(phone), w_word=0.0")

    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(args.seed)
    use_wandb = not args.no_wandb
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    assert not (args.use_prosody and args.utt_prosody), "pick one prosody mode"
    load_pros = args.use_prosody or args.utt_prosody

    # nguồn dữ liệu: bắt buộc HF v2 (cần msdd labels)
    from datasets import load_dataset
    dd = load_dataset(args.hf_dataset)
    def _src(split):
        return hf_split_to_npz_dict(dd[split], use_wavlm=args.use_wavlm, use_prosody=load_pros)
    print(f"Nạp HF: {args.hf_dataset} train={args.train} val={args.val} test={args.test} test2={args.test2}")
    train_src = _src(args.train)
    val_src = _src(args.val) if args.val in dd else None
    test_src = _src(args.test)
    test2_src = _src(args.test2) if args.test2 in dd else None

    tr = GOPTDataset(train_src, use_occ=args.use_occ, use_prosody=load_pros,
                     use_wavlm=args.use_wavlm, wavlm_dim=args.wavlm_dim, use_msdd=True,
                     feat_norm=args.feat_norm, wavlm_proj=args.wavlm_proj)
    def _mk(src):
        return GOPTDataset(src, feat_mean=tr.feat_mean, feat_std=tr.feat_std,
                           use_occ=args.use_occ, occ_mean=tr.occ_mean, occ_std=tr.occ_std,
                           use_prosody=load_pros, pros_mean=tr.pros_mean, pros_std=tr.pros_std,
                           use_wavlm=args.use_wavlm, wavlm_dim=args.wavlm_dim,
                           wavlm_pca=tr.wavlm_pca, wavlm_norm=tr.wavlm_norm, use_msdd=True,
                           feat_norm=args.feat_norm, wavlm_proj=args.wavlm_proj)
    te = _mk(test_src)
    va = _mk(val_src) if val_src is not None else te
    te2 = _mk(test2_src) if test2_src is not None else None
    if val_src is None:
        print("[WARN] không có val -> chọn best trên TEST (leak). Truyền --val để sửa.")

    gop_dim = tr.gop_dim
    wavlm_dim = tr.wavlm_width if args.use_wavlm else 0     # width thực (32 nếu PCA, 1024 nếu linear)
    enc_dim = gop_dim + (1 if args.use_occ else 0) + wavlm_dim
    input_dim = enc_dim + (8 if args.use_prosody else 0)
    prosody_dim = 8 if args.utt_prosody else 0
    score_scale = 100.0 if tr.is_scale_100 else 2.0

    # class-weight detection (sqrt-inverse-freq) tính trên train, chuẩn hóa mean=1
    vtr = tr.phn.numpy() >= 0
    mt = tr.msdd_type.numpy()[vtr]
    cnt = np.array([(mt == c).sum() for c in range(3)], float).clip(min=1)
    dw = (cnt.sum() / (3 * cnt)) ** args.det_wpow
    dw = (dw / dw.mean()).astype(np.float32)
    print(f"det counts={cnt.astype(int).tolist()} class_w={np.round(dw, 3).tolist()}")

    jcapt = {}
    if args.phono:
        from vh_gopt.core.phono import phono_buffer
        jcapt = dict(use_phono=True, phono_matrix=phono_buffer(tr.phone_list, 40))
    model = GOPTUnifiedForScoring(
        input_dim=input_dim, embed_dim=args.embed_dim, num_heads=args.heads, depth=args.depth,
        dropout=args.dropout, arch=args.arch, noise=args.noise, n_think=args.think,
        attn_pool=args.attn_pool, utt_prosody=args.utt_prosody, prosody_dim=prosody_dim,
        wavlm_dim=wavlm_dim, wavlm_fuse=args.wavlm_fuse, no_word_head=args.no_word_head,
        w_phn=args.w_phn, w_word=args.w_word, w_utt=args.w_utt,
        w_det=args.w_det, w_diag=args.w_diag, w_cons=args.w_cons,
        det_class_w=dw.tolist(), score_scale=score_scale, **jcapt).to(args.device)
    print(f"model=GOPTUnified input_dim={input_dim} embed={args.embed_dim} depth={args.depth} "
          f"params={sum(p.numel() for p in model.parameters())}")

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=256,
        learning_rate=args.lr, weight_decay=args.wd, adam_beta1=0.95, adam_beta2=0.999,
        lr_scheduler_type=("cosine" if args.sched == "cosine" else "constant"),
        warmup_steps=100, max_grad_norm=5.0,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model=f"eval_{args.best_metric}",
        greater_is_better=True, report_to=(["wandb"] if use_wandb else ["none"]),
        run_name=args.wandb_run, remove_unused_columns=False, label_names=LABEL_NAMES,
        dataloader_num_workers=0, seed=args.seed)

    optimizers = (None, None)
    if args.sched == "gopt":
        spe = math.ceil(len(tr) / args.bs)
        milestones = [e * spe for e in range(20, args.epochs, 5)]
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.95, 0.999))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 100.0, 1.0) * (0.5 ** sum(s >= ms for ms in milestones)))
        optimizers = (opt, sched)

    callbacks = []
    if args.early_stop_patience and args.early_stop_patience > 0:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience,
                                                early_stopping_threshold=args.early_stop_threshold))

    cm = make_compute_metrics(score_scale)
    trainer = Trainer(model=model, args=targs, train_dataset=tr, eval_dataset=va,
                      data_collator=collate, compute_metrics=cm,
                      optimizers=optimizers, callbacks=callbacks)
    trainer.train()

    def _eval(ds, prefix):
        m = trainer.evaluate(ds, metric_key_prefix=prefix)
        drop = {f"{prefix}_{s}" for s in ("loss", "runtime", "samples_per_second", "steps_per_second")}
        return {k.replace(f"{prefix}_", ""): v for k, v in m.items()
                if k.startswith(f"{prefix}_") and k not in drop}

    val_m = _eval(va, "val"); best = _eval(te, "test")
    all_test = {"test_unseen_speakers": best}
    print("\nVAL :", json.dumps({k: round(v, 4) for k, v in val_m.items()}, ensure_ascii=False))
    print("TEST(unseen_speakers):", json.dumps({k: round(v, 4) for k, v in best.items()}, ensure_ascii=False))
    if te2 is not None:
        t2 = _eval(te2, "test2"); all_test["test_unseen_prompts"] = t2
        print("TEST(unseen_prompts) :", json.dumps({k: round(v, 4) for k, v in t2.items()}, ensure_ascii=False))

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    _write_config(args, tr, val_m, all_test, score_scale)


def _write_config(args, tr, val_m, all_test, score_scale):
    wavlm_dim = args.wavlm_dim if args.use_wavlm else 0
    if args.use_wavlm and tr.wavlm_pca is not None:      # linear: không có PCA để lưu (projection nằm trong in_proj)
        mu, comp = tr.wavlm_pca; wm, ws = tr.wavlm_norm
        np.savez(os.path.join(args.out, "wavlm_pca.npz"),
                 mean=mu.numpy(), comp=comp.numpy(), norm_mean=wm.numpy(), norm_std=ws.numpy())
    elif args.use_wavlm:
        wm, ws = tr.wavlm_norm
        np.savez(os.path.join(args.out, "wavlm_norm.npz"), norm_mean=wm.numpy(), norm_std=ws.numpy())

    def _ser(v):                                          # tensor(perdim) -> list; float giữ nguyên
        return v.tolist() if hasattr(v, "tolist") else v
    cfg = {
        "arch": "GOPTUnified", "arch_variant": args.arch, "gop_dim": tr.gop_dim,
        "use_wavlm": args.use_wavlm, "wavlm_dim": wavlm_dim, "wavlm_fuse": args.wavlm_fuse,
        "wavlm_model": "microsoft/wavlm-large", "wavlm_layer": tr.wavlm_layer,
        "input_dim": tr.gop_dim + (1 if args.use_occ else 0) + wavlm_dim + (8 if args.use_prosody else 0),
        "embed_dim": args.embed_dim, "num_heads": args.heads, "depth": args.depth, "dropout": args.dropout,
        "max_len": 150, "n_phn_cls": 40, "use_occ": args.use_occ,
        "use_prosody": args.use_prosody, "utt_prosody": args.utt_prosody,
        "prosody_dim": 8 if args.utt_prosody else 0, "use_phono": args.phono,
        "n_think": args.think, "attn_pool": args.attn_pool, "no_word_head": args.no_word_head,
        "utt_heads": list(UTT_HEADS), "word_heads": list(WORD_HEADS), "phone_list": tr.phone_list,
        "det_classes": ["OK", "Sub", "Del"], "diag_classes": PHONE_NUM,
        "feat_norm": {"mean": _ser(tr.feat_mean), "std": _ser(tr.feat_std), "mode": args.feat_norm},
        "wavlm_proj": args.wavlm_proj,
        "label_scale": ({"phone": 1.0, "word": 1.0, "utt": 1.0, "to_100": 1.0}
                        if tr.is_scale_100 else {"phone": 1.0, "word": 5.0, "utt": 5.0, "to_100": 50.0}),
        "loss_weights": {"w_phn": args.w_phn, "w_word": args.w_word, "w_utt": args.w_utt,
                         "w_det": args.w_det, "w_diag": args.w_diag, "w_cons": args.w_cons},
        "hf_dataset": args.hf_dataset, "seed": args.seed, "best_metric": args.best_metric,
        "val_metrics": val_m, "all_test_metrics": all_test,
    }
    if args.use_occ:
        cfg["occ_norm"] = {"mean": tr.occ_mean, "std": tr.occ_std}
    if args.use_prosody or args.utt_prosody:
        cfg["pros_norm"] = {"mean": tr.pros_mean, "std": tr.pros_std}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(all_test, open(os.path.join(args.out, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    from vh_gopt.config import save_config
    save_config(vars(args), os.path.join(args.out, "config.yaml"))


if __name__ == "__main__":
    main()
