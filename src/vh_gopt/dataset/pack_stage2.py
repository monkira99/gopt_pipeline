#!/usr/bin/env python3
"""Dong goi dataset Stage 2 (GOPT Scoring + MSDD Stage 1) tu corpus snapshot.

Port tu pack_dataset_stage2.py (repo GOP_CTC_GOPT) voi cac thay doi:
  - Doc dau vao tu corpus snapshot di dong (xem vh_gopt.dataset.corpus):
    manifest + splits + vendor_eval/ (SpeechAce) + ss/ + iflytek/ + audio/.
  - Audio duoc resolve qua basename <corpus>/audio/<ID>.<ext> truoc, khong lech
    vao audio_path tuyet doi trong manifest goc.
  - MAC DINH trich feature CTC-GOP 80-d (KoelLabs/xlsr-english-01) + occupancy;
    chi dung --skip-gop khi can dong goi nhanh nhan-only (feat se la so 0,
    verify_dataset se CHAN truong hop nay khi push).
Cong thuc consensus giu nguyen goc: phone Median-of-3 / mean-of-2 (dz<=1) /
mask (nv<=1 hoac 2-vendor lech dz>1); word & utt head_consensus co trong so
c(3)=1.0, c(2)=0.6, c(1)=0.3; thang [0,100] quy doi ve khong gian SpeechSuper.
"""
import argparse
import json
import math
import os
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vh_gopt.config import add_config_arg, load_config
from vh_gopt.core import (
    PHONE2ID,
    align_sequence_dp,
    align_words_to_canon,
    detect_blank_id,
    get_canonical_word_phones,
    parse_iflytek_phones,
    parse_speechace_phones,
    parse_speechsuper_phones,
    IPA2ARPA_FULL,
)
from vh_gopt.dataset.corpus import check_layout, corpus_paths, load_manifest, resolve_audio

MAX_LEN = 150
UTT_HEADS = ("accuracy", "completeness", "fluency", "total")
WORD_HEADS = ("accuracy",)
# Z-score vendor tinh tren tap gold (giu nguyen goc de nhan khong doi)
ACE_MU, ACE_STD = 77.43, 32.07
SS_MU, SS_STD = 62.01, 40.70
IFLY_MU, IFLY_STD = 70.98, 27.91


def fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def head_consensus(zs, equiv, c_weights={3: 1.0, 2: 0.6, 1: 0.3}):
    """Consensus cho WORD/UTT head: median-of-3 / mean-of-2 (dz<=1) / keep nv=1 w=0.3."""
    nv = len(zs)
    if nv >= 3:
        zs.sort()
        d_rob = min(zs[1] - zs[0], zs[2] - zs[1])
        return float(np.median(equiv)), c_weights[3] / (1.0 + d_rob), nv
    elif nv == 2:
        d = abs(zs[0] - zs[1])
        if d <= 1.0:
            return float(np.mean(equiv)), c_weights[2] / (1.0 + d), nv
        else:
            return -1.0, 0.0, nv
    elif nv == 1:
        return float(equiv[0]), c_weights[1], nv
    return -1.0, 0.0, 0


def align_word_indices(canon_words, vendor_words):
    """Map chi so tu canonical -> chi so tu vendor (None neu khong khop)."""
    sm = SequenceMatcher(None, canon_words, vendor_words, autojunk=False)
    mapping = [None] * len(canon_words)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    return mapping


def pack_split(split_name, split_ids, manifest_dict, out_path, corpus_dir,
               model=None, processor=None, device="cpu", skip_gop=False, max_len=MAX_LEN):
    N = len(split_ids)
    print(f"\n--- PACK SPLIT: {split_name} ({N:,} mau) -> {out_path} ---")
    cp = corpus_paths(corpus_dir)

    GOP_DIM = 80 if model else 41
    feat = np.zeros((N, max_len, GOP_DIM), dtype=np.float32)
    occ = np.zeros((N, max_len), dtype=np.float32)
    phn = np.full((N, max_len), -1, dtype=np.int16)
    word_id = np.full((N, max_len), -1, dtype=np.int16)

    phone_label = np.full((N, max_len), -1.0, dtype=np.float32)
    phone_weight = np.zeros((N, max_len), dtype=np.float32)
    n_vendors = np.zeros((N, max_len), dtype=np.uint8)

    word_acc = np.full((N, max_len), -1.0, dtype=np.float32)
    word_weight = np.zeros((N, max_len), dtype=np.float32)

    utt_label = np.full((N, 4), -1.0, dtype=np.float32)
    utt_weight = np.zeros((N, 4), dtype=np.float32)
    utt_nv = np.zeros((N, 4), dtype=np.uint8)

    msdd_type = np.full((N, max_len), -1, dtype=np.int16)
    msdd_sub = np.full((N, max_len), -1, dtype=np.int16)

    ids_arr = np.array(split_ids, dtype="U32")
    texts_list = []
    skipped = 0

    if not skip_gop and model is not None:
        from vh_gopt.core.gop_feats_fast import extract_utt_feats_norm_fast
        from vh_gopt.core.koel_gop import map_phones_to_ids_koel
        blank_id = detect_blank_id(processor.tokenizer, model)

    for i, ID in enumerate(tqdm(split_ids, desc=f"Packing {split_name}")):
        r = manifest_dict.get(ID)
        if r is None:
            skipped += 1
            continue
        target = r["target_text"]
        texts_list.append(target)
        wav_path = resolve_audio(corpus_dir, r)
        try:
            ace = json.load(open(cp["ace_dir"] / f"{ID}.json"))["content"]["data"]
            ss = json.load(open(cp["ss_dir"] / f"{ID}.json"))
            ifl = json.load(open(cp["ifly_dir"] / f"{ID}.json"))
        except Exception:
            skipped += 1
            continue

        canon_words, canon_word_phones = get_canonical_word_phones(target)
        flat_canon = [p for w_phs in canon_word_phones for p in w_phs]

        ace_words, ace_wphs = parse_speechace_phones(ace)
        ss_words, ss_wphs = parse_speechsuper_phones(ss)
        ifl_words, ifl_wphs = parse_iflytek_phones(ifl)

        m_ace = align_words_to_canon(canon_words, ace_words, ace_wphs)
        m_ss = align_words_to_canon(canon_words, ss_words, ss_wphs)
        m_ifl = align_words_to_canon(canon_words, ifl_words, ifl_wphs)

        map_ace_w = align_word_indices(canon_words, ace_words)
        map_ss_w = align_word_indices(canon_words, ss_words)
        map_ifl_w = align_word_indices(canon_words, ifl_words)

        ace_raw_words = [float(w["quality_score"]) if fin(w.get("quality_score")) else None
                         for w in ace.get("word_score_list", [])]
        ss_raw_words = [float((w.get("scores") or {}).get("pronunciation")) if fin((w.get("scores") or {}).get("pronunciation")) else None
                        for w in ss.get("result", {}).get("words", [])]
        ifl_raw_words = [float(sc) if fin(sc) else None for sc in ifl.get("word_acc", [])]

        # 1. PHONE & WORD PROCESSING
        curr_p_idx = 0
        for w_idx, c_phs in enumerate(canon_word_phones):
            w_zs, w_eq = [], []
            va = map_ace_w[w_idx]
            if va is not None and va < len(ace_raw_words) and ace_raw_words[va] is not None:
                sc = ace_raw_words[va]; z = (sc - ACE_MU) / ACE_STD
                w_zs.append(z); w_eq.append(np.clip(SS_MU + z * SS_STD, 0, 100))
            vs = map_ss_w[w_idx]
            if vs is not None and vs < len(ss_raw_words) and ss_raw_words[vs] is not None:
                sc = ss_raw_words[vs]; z = (sc - SS_MU) / SS_STD
                w_zs.append(z); w_eq.append(float(sc))
            vi = map_ifl_w[w_idx]
            if vi is not None and vi < len(ifl_raw_words) and ifl_raw_words[vi] is not None:
                sc = ifl_raw_words[vi]; z = (sc - IFLY_MU) / IFLY_STD
                w_zs.append(z); w_eq.append(np.clip(SS_MU + z * SS_STD, 0, 100))

            w_val, w_wt, _ = head_consensus(w_zs, w_eq)

            a_aligned = align_sequence_dp(c_phs, m_ace[w_idx])
            s_aligned = align_sequence_dp(c_phs, m_ss[w_idx])
            i_aligned = align_sequence_dp(c_phs, m_ifl[w_idx])

            for p_sub_idx, c_p in enumerate(c_phs):
                if curr_p_idx >= max_len:
                    break

                phn[i, curr_p_idx] = PHONE2ID.get(c_p, -1)
                word_id[i, curr_p_idx] = w_idx

                # Word score broadcast xuong phone
                word_acc[i, curr_p_idx] = w_val
                word_weight[i, curr_p_idx] = w_wt

                # Phone consensus
                a_item = a_aligned[p_sub_idx]
                s_item = s_aligned[p_sub_idx]
                i_item = i_aligned[p_sub_idx]

                p_zs, p_eq = [], []
                if a_item and fin(a_item[1]):
                    z = (float(a_item[1]) - ACE_MU) / ACE_STD
                    p_zs.append(z); p_eq.append(np.clip(SS_MU + z * SS_STD, 0, 100))
                if s_item and fin(s_item[1]):
                    z = (float(s_item[1]) - SS_MU) / SS_STD
                    p_zs.append(z); p_eq.append(float(s_item[1]))
                if i_item and fin(i_item[1]):
                    z = (float(i_item[1]) - IFLY_MU) / IFLY_STD
                    p_zs.append(z); p_eq.append(np.clip(SS_MU + z * SS_STD, 0, 100))

                p_nv = len(p_zs)
                n_vendors[i, curr_p_idx] = p_nv

                if p_nv == 3:
                    p_zs.sort()
                    d_rob = min(p_zs[1] - p_zs[0], p_zs[2] - p_zs[1])
                    phone_label[i, curr_p_idx] = float(np.median(p_eq))
                    phone_weight[i, curr_p_idx] = 1.0 / (1.0 + d_rob)
                elif p_nv == 2:
                    d = abs(p_zs[0] - p_zs[1])
                    if d <= 1.0:
                        phone_label[i, curr_p_idx] = float(np.mean(p_eq))
                        phone_weight[i, curr_p_idx] = 0.6 / (1.0 + d)

                # MSDD Type & Sub
                a_ok = 1 if (a_item and a_item[1] is not None and a_item[1] >= 60.0) else 0
                s_ok = 1 if (s_item and s_item[1] is not None and s_item[1] >= 60.0) else 0
                i_ok = 1 if (i_item and i_item[1] is not None and i_item[1] >= 60.0) else 0
                s_rtype = s_item[2].get("readType", 0) if s_item else 0
                i_lab = i_item[2].get("label", "ok") if i_item else "ok"

                if s_rtype == 16 or i_lab == "del":
                    msdd_type[i, curr_p_idx] = 2  # Del
                elif (a_ok + s_ok + i_ok) >= 2:
                    msdd_type[i, curr_p_idx] = 0  # OK
                else:
                    msdd_type[i, curr_p_idx] = 1  # Sub

                # Am vi thay the (dong thuan chinh xac SpeechAce ∩ SpeechSuper)
                a_sub = a_item[2].get("sound_most_like") if a_item else None
                s_sub_ipa = s_item[2].get("sound_like") if s_item else None
                if a_sub and s_sub_ipa:
                    a_arpa = "".join(c for c in a_sub.upper() if c.isalpha())
                    s_arpa = IPA2ARPA_FULL.get(s_sub_ipa, s_sub_ipa.upper())
                    s_arpa = "".join(c for c in s_arpa if c.isalpha())
                    if a_arpa and s_arpa and a_arpa == s_arpa and a_arpa != c_p \
                            and msdd_type[i, curr_p_idx] == 1 and a_arpa in PHONE2ID:
                        msdd_sub[i, curr_p_idx] = PHONE2ID[a_arpa]

                curr_p_idx += 1

        # 2. UTTERANCE PROCESSING
        acc_zs, acc_eq = [], []
        v = (ace.get("speechace_score") or {}).get("pronunciation")
        if fin(v): z = (float(v) - ACE_MU) / ACE_STD; acc_zs.append(z); acc_eq.append(np.clip(SS_MU + z * SS_STD, 0, 100))
        v = (ss.get("result") or {}).get("pronunciation")
        if fin(v): z = (float(v) - SS_MU) / SS_STD; acc_zs.append(z); acc_eq.append(float(v))
        v = (ifl.get("utt") or {}).get("accuracy")
        if fin(v): z = (float(v) - IFLY_MU) / IFLY_STD; acc_zs.append(z); acc_eq.append(np.clip(SS_MU + z * SS_STD, 0, 100))
        val_a, w_a, nv_a = head_consensus(acc_zs, acc_eq)
        utt_label[i, 0] = val_a; utt_weight[i, 0] = w_a; utt_nv[i, 0] = nv_a

        # Head 1: Completeness (aux, chi SpeechSuper integrity)
        v = (ss.get("result") or {}).get("integrity", 100.0)
        utt_label[i, 1] = float(v) if fin(v) else 100.0
        utt_weight[i, 1] = 0.30; utt_nv[i, 1] = 1

        # Head 2: Fluency (SS + iFly)
        flu_zs, flu_eq = [], []
        v = (ss.get("result") or {}).get("fluency")
        if fin(v): z = (float(v) - SS_MU) / SS_STD; flu_zs.append(z); flu_eq.append(float(v))
        v = (ifl.get("utt") or {}).get("fluency")
        if fin(v): z = (float(v) - IFLY_MU) / IFLY_STD; flu_zs.append(z); flu_eq.append(np.clip(SS_MU + z * SS_STD, 0, 100))
        val_f, w_f, nv_f = head_consensus(flu_zs, flu_eq)
        utt_label[i, 2] = val_f; utt_weight[i, 2] = w_f; utt_nv[i, 2] = nv_f

        # Head 3: Total
        tot_zs, tot_eq = [], []
        v = (ace.get("speechace_score") or {}).get("overall")
        if fin(v): z = (float(v) - ACE_MU) / ACE_STD; tot_zs.append(z); tot_eq.append(np.clip(SS_MU + z * SS_STD, 0, 100))
        v = (ss.get("result") or {}).get("overall") or (ss.get("result") or {}).get("pronunciation")
        if fin(v): z = (float(v) - SS_MU) / SS_STD; tot_zs.append(z); tot_eq.append(float(v))
        v = (ifl.get("utt") or {}).get("total")
        if fin(v): z = (float(v) - IFLY_MU) / IFLY_STD; tot_zs.append(z); tot_eq.append(np.clip(SS_MU + z * SS_STD, 0, 100))
        val_t, w_t, nv_t = head_consensus(tot_zs, tot_eq)
        utt_label[i, 3] = val_t; utt_weight[i, 3] = w_t; utt_nv[i, 3] = nv_t

        # 3. STAGE 1 CTC GOP FEATURE EXTRACTION (80-d KoelLabs)
        if not skip_gop and model is not None and len(flat_canon) > 0 and wav_path is not None:
            import soundfile as sf
            try:
                wav_data, sr = sf.read(str(wav_path))
                if wav_data.ndim > 1:
                    wav_data = wav_data.mean(axis=1)
                if sr != 16000:
                    import librosa
                    wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=16000)

                iv = processor(wav_data, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                with torch.no_grad():
                    logits = model(iv).logits[0]
                post = torch.softmax(logits, dim=-1).cpu().type(torch.float64).T  # [P, T]

                labels_koel, _ = map_phones_to_ids_koel(flat_canon[:max_len], processor.tokenizer)
                labels_t = torch.tensor(labels_koel, dtype=torch.long)
                gop_res, occ_res = extract_utt_feats_norm_fast(post, labels_t, blank=blank_id, occ=True)
                n_p = min(len(labels_koel), max_len)
                feat[i, :n_p] = gop_res.numpy()[:n_p]
                if occ_res is not None:
                    occ[i, :n_p] = occ_res.numpy()[:n_p]
            except Exception as e:
                print(f"  [WARN] GOP extract fail {ID}: {e}")

    # Save .npz an toan allow_pickle=False
    np.savez_compressed(
        out_path,
        feat=feat, occ=occ, phn=phn, word_id=word_id,
        phone_label=phone_label, phone_weight=phone_weight, n_vendors=n_vendors,
        word_acc=word_acc, word_weight=word_weight,
        utt_label=utt_label, utt_weight=utt_weight, utt_nv=utt_nv,
        msdd_type=msdd_type, msdd_sub=msdd_sub,
        utt_heads=np.array(UTT_HEADS, dtype="U16"),
        word_heads=np.array(WORD_HEADS, dtype="U16"),
        phone_list=np.array(sorted(PHONE2ID), dtype="U8"),
        ids=ids_arr,
        texts=np.array(texts_list, dtype="U512"),
        N=np.int64(N), max_len=np.int64(max_len),
        scale=np.array("0-100", dtype="U8"),
    )
    pad = phn < 0
    nvp = int((~pad).sum())
    pv = float(((phone_label >= 0) & ~pad).sum()) / nvp if nvp else 0.0
    wv = float(((word_acc >= 0) & ~pad).sum()) / nvp if nvp else 0.0
    print(f"Xong {split_name}: {os.path.getsize(out_path)/1e6:.1f} MB | "
          f"phone_valid={pv*100:.2f}% word_valid={wv*100:.2f}% "
          f"msdd_gold={int((msdd_sub>=0).sum())} skipped={skipped}")


import torch  # noqa: E402  (dung o phan extraction)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_arg(ap)
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--manifest", default=None, help="override duong dan manifest jsonl")
    ap.add_argument("--splits", default=None, help="override file splits json")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model", default=None, help="acoustic model HF id cho CTC-GOP 80-d")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--skip-gop", action="store_true",
                    help="nhan-only, KHONG trich feature (feat = so 0; chi dung de test)")
    ap.add_argument("--limit", type=int, default=0, help="gioi han N mau/split (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpus_dir = args.corpus_dir or cfg["corpus_dir"]
    out_dir = Path(args.out_dir or cfg["out_dir"])
    max_len = args.max_len or int(cfg["max_len"])
    device = args.device or cfg["device"]
    model_id = args.model or cfg["model_id"]

    missing = check_layout(corpus_dir)
    if missing:
        raise SystemExit(f"Corpus snapshot thieu: {missing}\n"
                         f"Chay 'python -m vh_gopt.dataset.fetch_corpus' hoad snapshot bang tay.")

    cp = corpus_paths(corpus_dir)
    manifest_path = args.manifest or cp["manifest"]
    splits_path = args.splits or cp["splits"]
    manifest_dict = load_manifest(manifest_path)
    splits = json.load(open(splits_path, encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = processor = None
    if not args.skip_gop:
        from transformers import AutoModelForCTC, AutoProcessor
        print(f"Nap acoustic model: {model_id} tren {device}...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()

    total = 0
    for split_key in ["train", "val", "test_unseen_speakers", "test_unseen_prompts"]:
        split_ids = splits.get(split_key, [])
        if not split_ids:
            continue
        if args.limit > 0:
            split_ids = split_ids[:args.limit]
        total += len(split_ids)
        pack_split(split_key, split_ids, manifest_dict, str(out_dir / f"{split_key}.npz"),
                   corpus_dir, model, processor, device, args.skip_gop, max_len)
    print(f"\nHoan thanh {total:,} mau -> {out_dir}")


if __name__ == "__main__":
    main()
