#!/usr/bin/env python3
"""Trích xuất đầy đủ 100% các nhóm Feature trên GPU cho Baseline GOPT tốt nhất (Kèm Profiler chi tiết).

Bộ Feature được trích xuất gồm:
  1. KoelLabs-GOP 80-d (`feat` [N, 150, 80])
  2. Prosody 8-d (`dur` [N, 150] + `eng` [N, 150, 7])
  3. WavLM SSL (`wavlm` [N, 150, 1024] fp16)
"""
import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vh_gopt.config import add_config_arg, load_config
from vh_gopt.core import (
    PHONE_LIST,
    detect_blank_id,
    phone_prosody_from_segs,
    phone_segments,
)
from vh_gopt.core.gop_feats_fast import extract_utt_feats_norm_fast
from vh_gopt.core.koel_gop import map_phones_to_ids_koel

SPLITS = ["train", "val", "test_unseen_speakers", "test_unseen_prompts"]


def sync_cuda(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def print_cuda_diagnostics(device, koel_model, wl_model=None, use_fp16=True, batch_size=16, num_workers=2):
    """Kiểm tra và in chi tiết trạng thái nạp model vào GPU/CUDA."""
    print("\n" + "=" * 70)
    print("🔍 [CUDA & ENGINE DIAGNOSTICS] KIỂM TRA THIẾT BỊ VÀ TRẠNG THÁI TỐI ƯU:")
    print("=" * 70)
    
    cuda_avail = torch.cuda.is_available()
    print(f" • PyTorch Version       : {torch.__version__}")
    print(f" • CUDA Khả dụng          : {'✅ CÓ (CUDA Available)' if cuda_avail else '❌ KHÔNG (Chạy CPU)'}")
    
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f" • GPU Tên                : 🚀 {gpu_name}")
        print(f" • VRAM Bộ nhớ           : {vram_alloc:.2f} GB / {vram_total:.2f} GB (Allocated / Total)")
        print(f" • Tensor Cores (FP16)   : {'✅ BẬT (Half Precision)' if use_fp16 else '❌ TẮT (FP32)'}")
    
    koel_dev = next(koel_model.parameters()).device
    koel_dtype = next(koel_model.parameters()).dtype
    print(f" • KoelLabs Acoustic CTC : Vị trí = {koel_dev} | Dtype = {koel_dtype}")
    
    if wl_model is not None:
        wl_dev = next(wl_model.parameters()).device
        wl_dtype = next(wl_model.parameters()).dtype
        num_layers = len(wl_model.encoder.layers)
        print(f" • WavLM SSL Model       : Vị trí = {wl_dev} | Dtype = {wl_dtype} | Layers = {num_layers} (Truncated 0..12)")

    print(f" • Pipeline Configuration: Batch Size = {batch_size} | CPU Background Workers = {num_workers}")
    print("=" * 70 + "\n")


def fast_resample(wav_np, orig_sr, target_sr=16000):
    """Resample waveform cực nhanh bằng PyTorch 1D interpolation."""
    if orig_sr == target_sr:
        return wav_np.astype(np.float32)
    t_wav = torch.from_numpy(wav_np).float().view(1, 1, -1)
    target_len = int(round(len(wav_np) * target_sr / orig_sr))
    resampled = F.interpolate(t_wav, size=target_len, mode="linear", align_corners=False)
    return resampled.squeeze().numpy()


class AudioExtractionDataset(Dataset):
    """CPU Background Producer: Giải mã audio và chuẩn bị mảng nhãn bất đồng bộ."""
    def __init__(self, ds_split, limit=0, max_len=150):
        self.ds = ds_split if limit == 0 else ds_split.select(range(min(limit, len(ds_split))))
        self.max_len = max_len
        self.phone_list = self.ds[0].get("phone_list") or PHONE_LIST

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        rid = str(ex["id"])
        text = str(ex["text"])

        # 1. Giải mã audio trên CPU worker
        a_dict = ex["audio"]
        if a_dict.get("bytes") is not None:
            wav_data, sr = sf.read(io.BytesIO(a_dict["bytes"]))
        elif a_dict.get("path") is not None and os.path.exists(a_dict["path"]):
            wav_data, sr = sf.read(a_dict["path"])
        else:
            wav_data, sr = np.zeros(16000, dtype=np.float32), 16000

        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        if sr != 16000:
            wav_data = fast_resample(wav_data, sr, 16000)

        # 2. Chuẩn bị chuỗi âm vị canonical
        valid_phn_ids = [int(p) for p in ex["phn"] if p >= 0][:self.max_len]
        canon_phones = [self.phone_list[pid] for pid in valid_phn_ids]

        return {
            "idx": idx,
            "id": rid,
            "text": text,
            "wav": wav_data.astype(np.float32),
            "canon_phones": canon_phones,
            "phn": np.array(ex["phn"][:self.max_len], dtype=np.int16),
            "phone_label": np.array(ex["phone_label"][:self.max_len], dtype=np.float32),
            "phone_weight": np.array(ex["phone_weight"][:self.max_len], dtype=np.float32),
            "n_vendors": np.array(ex["n_vendors"][:self.max_len], dtype=np.uint8),
            "word_id": np.array(ex["word_id"][:self.max_len], dtype=np.int16),
            "word_acc": np.array(ex["word_acc"][:self.max_len], dtype=np.float32),
            "word_weight": np.array(ex["word_weight"][:self.max_len], dtype=np.float32),
            "utt_label": np.array(ex["utt_label"], dtype=np.float32),
            "utt_weight": np.array(ex["utt_weight"], dtype=np.float32),
            "utt_nv": np.array(ex["utt_nv"], dtype=np.uint8),
            "msdd_type": np.array(ex["msdd_type"][:self.max_len], dtype=np.int16),
            "msdd_sub": np.array(ex["msdd_sub"][:self.max_len], dtype=np.int16),
        }


def load_input_dataset(args, cfg):
    from datasets import Audio, load_dataset, load_from_disk

    if args.dataset_dir and os.path.exists(args.dataset_dir):
        print(f"Nạp dataset từ thư mục local: {args.dataset_dir}")
        ds = load_from_disk(args.dataset_dir)
    elif os.path.exists("data/gopt_vh_gold_arrow"):
        print("Tìm thấy local cache: data/gopt_vh_gold_arrow -> nạp trực tiếp...")
        ds = load_from_disk("data/gopt_vh_gold_arrow")
    else:
        repo = args.dataset_repo or cfg.get("hf_dataset_repo") or "tiennguyenbnbk/gopt-vh-gold"
        print(f"Tải dataset từ HuggingFace Hub: {repo} ...")
        ds = load_dataset(repo)

    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def extract_split_features(split_name, ds_split, out_path,
                           koel_model, koel_processor, blank_id,
                           wl_model, wl_fe, wl_layer,
                           device="cuda", limit=0, max_len=150,
                           use_wavlm=True, use_prosody=True, use_fp16=True,
                           batch_size=16, num_workers=2):
    dataset = AudioExtractionDataset(ds_split, limit=limit, max_len=max_len)
    N = len(dataset)
    print(f"============================================================")
    print(f"--- TRÍCH XUẤT FEATURE: {split_name} ({N:,} mẫu | Batch Size = {batch_size}) -> {out_path} ---")
    print(f"============================================================")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: b,
        pin_memory=device.startswith("cuda"),
    )

    GOP_DIM = 80
    feat = np.zeros((N, max_len, GOP_DIM), dtype=np.float32)
    occ = np.zeros((N, max_len), dtype=np.float32)
    dur = np.zeros((N, max_len), dtype=np.float32)
    eng = np.zeros((N, max_len, 7), dtype=np.float32)
    wavlm = np.zeros((N, max_len, 1024), dtype=np.float16)

    phn = np.full((N, max_len), -1, dtype=np.int16)
    phone_label = np.full((N, max_len), -1.0, dtype=np.float32)
    phone_weight = np.zeros((N, max_len), dtype=np.float32)
    n_vendors = np.zeros((N, max_len), dtype=np.uint8)

    word_id = np.full((N, max_len), -1, dtype=np.int16)
    word_acc = np.full((N, max_len), -1.0, dtype=np.float32)
    word_weight = np.zeros((N, max_len), dtype=np.float32)

    utt_label = np.full((N, 4), -1.0, dtype=np.float32)
    utt_weight = np.zeros((N, 4), dtype=np.float32)
    utt_nv = np.zeros((N, 4), dtype=np.uint8)

    msdd_type = np.full((N, max_len), -1, dtype=np.int16)
    msdd_sub = np.full((N, max_len), -1, dtype=np.int16)

    ids_list = [""] * N
    texts_list = [""] * N

    # Profiling accumulators
    prof = {
        "data_wait": 0.0,
        "koel_fwd": 0.0,
        "wavlm_fwd": 0.0,
        "gop_dp": 0.0,
        "alignment_prosody": 0.0,
        "wavlm_pool": 0.0,
        "total_utt_count": 0,
    }

    t0 = time.time()
    curr_idx = 0
    pbar = tqdm(total=N, desc=f"Extracting {split_name}", unit="utt")
    batch_idx = 0

    t_data_start = time.time()
    for batch in loader:
        t_data_end = time.time()
        prof["data_wait"] += (t_data_end - t_data_start)

        B = len(batch)
        batch_wavs = [item["wav"] for item in batch]

        # ---- PHASE 1: BATCHED GPU NEURAL INFERENCE (KOELLABS CTC) ----
        sync_cuda(device)
        t_k0 = time.time()
        iv = koel_processor(batch_wavs, sampling_rate=16000, padding=True, return_tensors="pt").input_values.to(device)
        if use_fp16 and device.startswith("cuda"):
            iv = iv.half()

        with torch.inference_mode():
            batch_koel_logits = koel_model(iv).logits  # [B, T_max, 80]
        sync_cuda(device)
        prof["koel_fwd"] += (time.time() - t_k0)

        # ---- PHASE 2: BATCHED GPU NEURAL INFERENCE (TRUNCATED WAVLM) ----
        batch_wl_hs = None
        if use_wavlm and wl_model is not None:
            sync_cuda(device)
            t_w0 = time.time()
            wf = wl_fe(batch_wavs, sampling_rate=16000, padding=True, return_tensors="pt").input_values.to(device)
            if use_fp16 and device.startswith("cuda"):
                wf = wf.half()

            with torch.inference_mode():
                wl_out = wl_model(wf, output_hidden_states=True)
                batch_wl_hs = wl_out.hidden_states[wl_layer]  # [B, Tw_max, 1024]
            sync_cuda(device)
            prof["wavlm_fwd"] += (time.time() - t_w0)

        # ---- PHASE 3: VECTORIZED GPU GOP & ALIGNMENT PROCESSING ----
        for b in range(B):
            item = batch[b]
            i = curr_idx + b
            rid = item["id"]
            ids_list[i] = rid
            texts_list[i] = item["text"]

            phn[i] = item["phn"]
            phone_label[i] = item["phone_label"]
            phone_weight[i] = item["phone_weight"]
            n_vendors[i] = item["n_vendors"]

            word_id[i] = item["word_id"]
            word_acc[i] = item["word_acc"]
            word_weight[i] = item["word_weight"]

            utt_label[i] = item["utt_label"]
            utt_weight[i] = item["utt_weight"]
            utt_nv[i] = item["utt_nv"]

            msdd_type[i] = item["msdd_type"]
            msdd_sub[i] = item["msdd_sub"]

            canon_phones = item["canon_phones"]
            S = len(canon_phones)
            wav_data = item["wav"]
            T_actual = max(1, int(len(wav_data) // 320))

            if S > 0:
                labels_koel, _ = map_phones_to_ids_koel(canon_phones, koel_processor.tokenizer)
                labels_t = torch.tensor(labels_koel, dtype=torch.long, device=device)
                koel_logits = batch_koel_logits[b, :T_actual]

                # 3A. GOP Dynamic Programming
                sync_cuda(device)
                t_g0 = time.time()
                post = torch.softmax(koel_logits.float(), dim=-1).type(torch.float64).T
                gop_res, occ_res = extract_utt_feats_norm_fast(post, labels_t, blank=blank_id, occ=True)
                feat[i, :S] = gop_res.numpy()[:S]
                if occ_res is not None:
                    occ[i, :S] = occ_res.numpy()[:S]
                sync_cuda(device)
                prof["gop_dp"] += (time.time() - t_g0)
                # 3B. Single-pass Viterbi Alignment & Prosody
                t_a0 = time.time()
                logp = koel_logits.float().log_softmax(-1).cpu().double()
                labels_cpu = labels_t.cpu()
                segs, T_ctc, _ = phone_segments(logp, labels_cpu, blank=blank_id)

                if use_prosody and segs is not None:
                    d_res, e_res = phone_prosody_from_segs(segs, T_ctc, wav_data, S)
                    dur[i, :S] = d_res[:S]
                    eng[i, :S] = e_res[:S]
                prof["alignment_prosody"] += (time.time() - t_a0)

                # 3C. WavLM Mean-Pooling
                if use_wavlm and batch_wl_hs is not None and segs is not None:
                    t_p0 = time.time()
                    Tw_actual = max(1, int(len(wav_data) // 320))
                    hs = batch_wl_hs[b, :Tw_actual].float().cpu()  # [Tw, 1024]
                    Tw = hs.shape[0]
                    ratio = Tw / max(T_ctc, 1)
                    for k, (a, b_seg) in enumerate(segs):
                        if k >= max_len:
                            break
                        a2, b2 = int(a * ratio), max(int(b_seg * ratio), int(a * ratio) + 1)
                        wavlm[i, k] = hs[a2:min(b2, Tw)].mean(0).numpy().astype(np.float16)
                    prof["wavlm_pool"] += (time.time() - t_p0)

        curr_idx += B
        prof["total_utt_count"] += B
        batch_idx += 1
        pbar.update(B)

        # In bảng Profiler chi tiết mỗi 5 batch đầu tiên và sau đó mỗi 20 batch
        if (batch_idx <= 5) or (batch_idx % 20 == 0):
            c = prof["total_utt_count"]
            t_data_ms = (prof["data_wait"] / c) * 1000
            t_koel_ms = (prof["koel_fwd"] / c) * 1000
            t_wl_ms = (prof["wavlm_fwd"] / c) * 1000
            t_gop_ms = (prof["gop_dp"] / c) * 1000
            t_align_ms = (prof["alignment_prosody"] / c) * 1000
            t_pool_ms = (prof["wavlm_pool"] / c) * 1000
            t_sum_ms = t_data_ms + t_koel_ms + t_wl_ms + t_gop_ms + t_align_ms + t_pool_ms

            pbar.write(
                f"⏱️ [PROFILER | {curr_idx}/{N} utts] Thời gian TB: {t_sum_ms:.1f} ms/utt ({t_sum_ms/1000:.3f}s) | "
                f"Data(CPU): {t_data_ms:.1f}ms ({(t_data_ms/t_sum_ms)*100:.1f}%) | "
                f"Koel(GPU): {t_koel_ms:.1f}ms ({(t_koel_ms/t_sum_ms)*100:.1f}%) | "
                f"WavLM(GPU): {t_wl_ms:.1f}ms ({(t_wl_ms/t_sum_ms)*100:.1f}%) | "
                f"GOP(DP): {t_gop_ms:.1f}ms ({(t_gop_ms/t_sum_ms)*100:.1f}%) | "
                f"Align+Pros: {t_align_ms:.1f}ms ({(t_align_ms/t_sum_ms)*100:.1f}%)"
            )

        t_data_start = time.time()

    pbar.close()
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    phone_list_save = np.array(ds_split[0].get("phone_list") or PHONE_LIST, dtype="U8")
    utt_heads_save = np.array(ds_split[0].get("utt_heads") or ["accuracy", "completeness", "fluency", "total"], dtype="U16")
    word_heads_save = np.array(ds_split[0].get("word_heads") or ["accuracy"], dtype="U16")
    scale_save = np.array(ds_split[0].get("scale") or "0-100", dtype="U8")

    save_dict = {
        "feat": feat,
        "occ": occ,
        "dur": dur,
        "eng": eng,
        "wavlm": wavlm,
        "wavlm_layer": np.int64(wl_layer),
        "phn": phn,
        "word_id": word_id,
        "phone_label": phone_label,
        "phone_weight": phone_weight,
        "n_vendors": n_vendors,
        "word_acc": word_acc,
        "word_weight": word_weight,
        "utt_label": utt_label,
        "utt_weight": utt_weight,
        "utt_nv": utt_nv,
        "msdd_type": msdd_type,
        "msdd_sub": msdd_sub,
        "utt_heads": utt_heads_save,
        "word_heads": word_heads_save,
        "phone_list": phone_list_save,
        "ids": np.array(ids_list, dtype="U32"),
        "texts": np.array(texts_list, dtype="U512"),
        "N": np.int64(N),
        "max_len": np.int64(max_len),
        "scale": scale_save,
    }

    np.savez_compressed(out_path, **save_dict)
    dt = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    print(f">> HOÀN TẤT {split_name}: {N} mẫu | {size_mb:.1f} MB | {dt:.1f}s ({dt/max(N,1):.2f}s/utt) -> {out_path}\n")


def push_extracted_features_to_hub(npz_dir, splits, repo, public=False):
    from datasets import Array2D, Dataset, DatasetDict, Features, Sequence, Value
    from huggingface_hub import HfApi

    print(f"\n>> Đang đóng gói và đẩy dataset có đủ Feature lên HuggingFace: {repo} ...")
    dd = {}
    for s in splits:
        p = os.path.join(npz_dir, f"{s}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        N, L = int(z["N"]), int(z["max_len"])
        seq = lambda dt, n=L: Sequence(Value(dt), length=n)  # noqa: E731

        feats_dict = {
            "id": Value("string"),
            "text": Value("string"),
            "phone_list": Sequence(Value("string"), length=39),
            "utt_heads": Sequence(Value("string"), length=4),
            "word_heads": Sequence(Value("string"), length=1),
            "scale": Value("string"),
            "max_len": Value("int32"),
            "phn": seq("int16"),
            "phone_label": seq("float32"),
            "phone_weight": seq("float32"),
            "n_vendors": seq("uint8"),
            "word_id": seq("int16"),
            "word_acc": seq("float32"),
            "word_weight": seq("float32"),
            "msdd_type": seq("int16"),
            "msdd_sub": seq("int16"),
            "utt_label": Sequence(Value("float32"), length=4),
            "utt_weight": Sequence(Value("float32"), length=4),
            "utt_nv": Sequence(Value("uint8"), length=4),
            "feat": Array2D(shape=(L, 80), dtype="float32"),
            "occ": seq("float32"),
            "dur": seq("float32"),
            "eng": Array2D(shape=(L, 7), dtype="float32"),
        }
        if "wavlm" in z and z["wavlm"].shape[-1] == 1024:
            feats_dict["wavlm"] = Array2D(shape=(L, 1024), dtype="float16")
            feats_dict["wavlm_layer"] = Value("int32")

        cols_data = {
            "id": [str(x) for x in z["ids"]],
            "text": [str(x) for x in z["texts"]],
            "phone_list": [[str(x) for x in z["phone_list"]]] * N,
            "utt_heads": [[str(x) for x in z["utt_heads"]]] * N,
            "word_heads": [[str(x) for x in z["word_heads"]]] * N,
            "scale": [str(z["scale"])] * N,
            "max_len": [L] * N,
        }
        for k in ["phn", "phone_label", "phone_weight", "n_vendors",
                  "word_id", "word_acc", "word_weight",
                  "msdd_type", "msdd_sub", "utt_label", "utt_weight", "utt_nv",
                  "feat", "occ", "dur", "eng"]:
            arr = z[k]
            cols_data[k] = list(arr) if arr.ndim > 1 else arr.tolist()

        if "wavlm" in feats_dict:
            cols_data["wavlm"] = list(z["wavlm"])
            cols_data["wavlm_layer"] = [int(z.get("wavlm_layer", 12))] * N

        ds = Dataset.from_dict(cols_data, features=Features(feats_dict))
        dd[s] = ds
        print(f"  [HF pack] {s}: {len(ds):,} rows")

    dsdict = DatasetDict(dd)
    private = not public
    dsdict.push_to_hub(repo, private=private)

    card = f"""---
language:
- en
pretty_name: "VuiHoc GOPT Gold Features (GOP-80d + Prosody-8d + WavLM-1024d)"
tags:
- goodness-of-pronunciation
- pronunciation-assessment
- wavlm
- gopt
size_categories:
- 1K<n<10K
---

# VuiHoc GOPT Gold Features (Đã trích xuất đầy đủ Feature)

Dataset GOPT VuiHoc Gold (6,361 câu) đã được trích xuất sẵn toàn bộ Feature trên GPU:
- `feat` [150, 80]: KoelLabs-GOP (log-posterior ratio normalized)
- `occ` [150]: Soft occupancy
- `dur` [150] + `eng` [150, 7]: Prosody 8-d (Viterbi forced alignment + RMSE energy)
- `wavlm` [150, 1024]: WavLM-large Layer 12 pooled per-phone (fp16)
- Nhãn: `phn`, `phone_label`, `word_acc`, `utt_label`, `msdd_type`, `msdd_sub`

## Load trực tiếp để train

```python
from datasets import load_dataset
ds = load_dataset("{repo}")
```
"""
    api = HfApi()
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    print(f">> ĐÃ PUSH THÀNH CÔNG FEATURE DATASET LÊN HF: https://huggingface.co/datasets/{repo} (private={private})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_arg(ap)
    ap.add_argument("--dataset-repo", default=None, help="Repo HF (mặc định: tiennguyenbnbk/gopt-vh-gold)")
    ap.add_argument("--dataset-dir", default=None, help="Hoặc thư mục DatasetDict local")
    ap.add_argument("--out-dir", default="data/gopt_vh_scripted_gold", help="Thư mục xuất .npz")
    ap.add_argument("--acoustic-model", default="KoelLabs/xlsr-english-01")
    ap.add_argument("--wavlm-model", default="microsoft/wavlm-large")
    ap.add_argument("--wavlm-layer", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size chạy GPU (mặc định: 16)")
    ap.add_argument("--num-workers", type=int, default=2, help="Số worker CPU giải mã âm thanh (Colab tối ưu = 2)")
    ap.add_argument("--no-fp16", action="store_true", help="Tắt chế độ FP16 trên GPU")
    ap.add_argument("--no-wavlm", action="store_true", help="Bỏ qua trích xuất WavLM")
    ap.add_argument("--no-prosody", action="store_true", help="Bỏ qua trích xuất Prosody")
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số mẫu/split để smoke test (0 = tất cả)")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--push-repo", default=None, help="Repo HF để đẩy toàn bộ dataset có đủ Feature lên (vd: <org>/gopt-vh-gold-features)")
    ap.add_argument("--public", action="store_true", help="Công khai repo HF (mặc định private)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_fp16 = not args.no_fp16 and args.device.startswith("cuda")

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True

    ds = load_input_dataset(args, cfg)

    # 1. Nạp Acoustic Model (KoelLabs CTC)
    from transformers import AutoModelForCTC, AutoProcessor
    print(f"\n[1/2] Nạp Acoustic Model: {args.acoustic_model} ...")
    koel_proc = AutoProcessor.from_pretrained(args.acoustic_model)
    koel_model = AutoModelForCTC.from_pretrained(args.acoustic_model).to(args.device).eval()
    if use_fp16:
        koel_model = koel_model.half()
    blank_id = detect_blank_id(koel_proc.tokenizer, koel_model)

    # 2. Nạp WavLM Model (Cắt ngắn đúng Layer 12 để loại bỏ 50% tính toán thừa)
    wl_model, wl_fe = None, None
    use_wavlm = not args.no_wavlm
    if use_wavlm:
        from transformers import AutoFeatureExtractor, WavLMModel
        print(f"\n[2/2] Nạp WavLM SSL Model: {args.wavlm_model} (Layer {args.wavlm_layer}) ...")
        wl_fe = AutoFeatureExtractor.from_pretrained(args.wavlm_model)
        wl_model = WavLMModel.from_pretrained(args.wavlm_model, attn_implementation="eager").to(args.device).eval()
        # Truncate WavLM encoder tới Layer 12
        wl_model.encoder.layers = wl_model.encoder.layers[:args.wavlm_layer + 1]
        if use_fp16:
            wl_model = wl_model.half()

    # 3. In bảng kiểm tra CUDA Diagnostics
    print_cuda_diagnostics(args.device, koel_model, wl_model, use_fp16, args.batch_size, args.num_workers)

    # 4. Chạy trích xuất từng Split với Batched Pipeline
    for s in splits:
        if s not in ds:
            print(f"[SKIP] Không tìm thấy split '{s}' trong dataset")
            continue
        out_file = out_dir / f"{s}.npz"
        extract_split_features(
            split_name=s,
            ds_split=ds[s],
            out_path=str(out_file),
            koel_model=koel_model,
            koel_processor=koel_proc,
            blank_id=blank_id,
            wl_model=wl_model,
            wl_fe=wl_fe,
            wl_layer=args.wavlm_layer,
            device=args.device,
            limit=args.limit,
            max_len=args.max_len,
            use_wavlm=use_wavlm,
            use_prosody=not args.no_prosody,
            use_fp16=use_fp16,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print("\n============================================================")
    print(f"TRÍCH XUẤT TOÀN BỘ FEATURE HOÀN TẤT! File .npz sẵn sàng tại: {out_dir}")
    print("Có thể chạy huấn luyện ngay:")
    print(f"  ./run.sh train-stage2 --config configs/stage2/baseline_wavlm32.yaml")
    print("============================================================")

    if args.push_repo:
        push_extracted_features_to_hub(
            npz_dir=str(out_dir),
            splits=splits,
            repo=args.push_repo,
            public=args.public,
        )


if __name__ == "__main__":
    main()
