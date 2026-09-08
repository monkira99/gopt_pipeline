#!/usr/bin/env python3
"""Điền cột `occ` (soft occupancy) vào dataset features rồi push lên HF.

Bối cảnh: extract_utt_feats_fb TRƯỚC ĐÂY trả occ=None nên extract_features.py không
bao giờ điền occ -> cột occ trong repo features TOÀN 0 -> bật --use-occ chia cho std=0
-> loss NaN. Bug đã sửa ở gop_feats_fast.py (occ = sum_t gamma[2i+1], tái dùng LA/LB).

Script này CHỈ tính lại occ từ audio gold-arrow qua Koel CTC + forward-backward đã sửa,
căn theo `id`, THAY cột occ trong repo features, GIỮ NGUYÊN feat/dur/eng/wavlm (nên các
model đã train trên các cột kia vẫn so sánh được). Koel-only: KHÔNG chạy lại WavLM.

Chạy (pod có GPU):
  HF_TOKEN=... python -m vh_gopt.dataset.fill_occ \
      --push-repo tiennguyenbnbk/gopt-vh-gold-features-v2 --batch-size 16
"""
import argparse
import io
import os

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from vh_gopt.core import PHONE_LIST, detect_blank_id
from vh_gopt.core.gop_feats_fast import extract_utt_feats_fb
from vh_gopt.core.koel_gop import map_phones_to_ids_koel

SPLITS = ["train", "val", "test_unseen_speakers", "test_unseen_prompts"]


def fast_resample(wav, sr, tgt=16000):
    if sr == tgt:
        return wav
    x = torch.tensor(wav, dtype=torch.float32).view(1, 1, -1)
    n = int(round(wav.shape[0] * tgt / sr))
    return torch.nn.functional.interpolate(x, size=n, mode="linear", align_corners=False).view(-1).numpy()


def decode_wav(a):
    """Giải mã bằng soundfile từ bytes/path -> né phụ thuộc torchcodec của HF Audio."""
    if a.get("bytes") is not None:
        w, sr = sf.read(io.BytesIO(a["bytes"]))
    elif a.get("path") and os.path.exists(a["path"]):
        w, sr = sf.read(a["path"])
    else:
        return np.zeros(16000, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != 16000:
        w = fast_resample(w, sr, 16000)
    return w.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="tiennguyenbnbk/gopt-vh-gold", help="Repo có audio + phn")
    ap.add_argument("--features", default="tiennguyenbnbk/gopt-vh-gold-features-v2", help="Repo features cần vá occ")
    ap.add_argument("--push-repo", default="tiennguyenbnbk/gopt-vh-gold-features-v2", help="Repo đích để push (mặc định ghi đè features)")
    ap.add_argument("--acoustic-model", default="KoelLabs/xlsr-english-01")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn/utt mỗi split để smoke test (0=all)")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="Chỉ tính + báo cáo, không push (kiểm thử)")
    args = ap.parse_args()

    use_fp16 = not args.no_fp16 and args.device.startswith("cuda")
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    print(f"Nạp gold (audio): {args.gold}")
    gold = load_dataset(args.gold)
    gold = gold.cast_column("audio", Audio(decode=False))   # {bytes,path} thô, dùng soundfile
    print(f"Nạp features (cần vá occ): {args.features}")
    feats = load_dataset(args.features)

    from transformers import AutoModelForCTC, AutoProcessor
    print(f"Nạp Koel CTC: {args.acoustic_model} (device={args.device}, fp16={use_fp16})")
    proc = AutoProcessor.from_pretrained(args.acoustic_model)
    kw = {}
    if hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.device.startswith("cuda"):
        kw["attn_implementation"] = "sdpa"
    model = AutoModelForCTC.from_pretrained(args.acoustic_model, **kw).to(args.device).eval()
    if use_fp16:
        model = model.half()
    blank_id = detect_blank_id(proc.tokenizer, model)
    out_len = getattr(model, "_get_feat_extract_output_lengths", None)
    print(f"blank_id={blank_id}")

    for sp in SPLITS:
        if sp not in feats:
            continue
        g = gold[sp]
        n = len(g) if args.limit == 0 else min(args.limit, len(g))
        phone_list = g[0].get("phone_list") or PHONE_LIST
        occ_by_id = {}
        n_empty = 0

        gv = g.select(range(n))
        pbar = tqdm(range(0, n, args.batch_size), desc=f"occ {sp}", unit="batch")
        for start in pbar:
            rows = gv[start:start + args.batch_size]          # dict-of-columns
            B = len(rows["id"])
            wavs = [decode_wav(rows["audio"][b]) for b in range(B)]
            k_inp = proc(wavs, sampling_rate=16000, padding=True, return_tensors="pt")
            iv = k_inp.input_values.to(args.device)
            if use_fp16:
                iv = iv.half()
            mask = k_inp.attention_mask.to(args.device, dtype=torch.long) if getattr(k_inp, "attention_mask", None) is not None else None
            with torch.inference_mode():
                logits = model(iv, attention_mask=mask).logits    # [B,T,P]
            logits = logits.float().cpu()

            for b in range(B):
                rid = str(rows["id"][b])
                phn = rows["phn"][b]
                valid = [int(p) for p in phn if p >= 0][:args.max_len]
                S = len(valid)
                occ_full = np.zeros(args.max_len, dtype=np.float32)
                if S > 0:
                    canon = [phone_list[p] for p in valid]
                    labels, _ = map_phones_to_ids_koel(canon, proc.tokenizer)
                    labels_cpu = torch.tensor(labels, dtype=torch.long)
                    T_act = int(out_len(torch.tensor(len(wavs[b]))).item()) if out_len is not None else max(1, len(wavs[b]) // 320)
                    T_act = min(T_act, logits.shape[1])
                    post = torch.softmax(logits[b, :T_act], dim=-1).T   # [P,T]
                    _, occ = extract_utt_feats_fb(post, labels_cpu, blank=blank_id)
                    occ_full[:S] = occ.numpy()[:S]
                else:
                    n_empty += 1
                occ_by_id[rid] = occ_full.tolist()
            pbar.set_postfix(empty=n_empty)

        # Thay cột occ trong split features theo đúng thứ tự id của nó
        f = feats[sp]
        ids = f["id"]
        miss = [i for i in ids if str(i) not in occ_by_id]
        if miss:
            raise SystemExit(f"[{sp}] {len(miss)} id trong features không có ở gold (vd {miss[:3]}) — không thể căn.")
        occ_col = [occ_by_id[str(i)] for i in ids]
        # sanity: std trên vị trí valid phải > 0 (occ không còn toàn 0)
        arr = np.array(occ_col, dtype=np.float32)
        nz = float((arr > 0).mean())
        print(f"[{sp}] occ filled: rows={len(occ_col)} frac_nonzero={nz:.3f} "
              f"mean={arr[arr>0].mean():.3f} max={arr.max():.1f} empty_utt={n_empty}")
        if "occ" in f.column_names:
            f = f.remove_columns("occ")
        feats[sp] = f.add_column("occ", occ_col)

    if args.no_push:
        print("--no-push: bỏ qua đẩy HF.")
        return
    print(f"Push -> {args.push_repo} (private={not args.public})")
    feats.push_to_hub(args.push_repo, private=not args.public)
    print("XONG.")


if __name__ == "__main__":
    main()
