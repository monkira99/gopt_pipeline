#!/usr/bin/env python3
"""
Đánh giá chi tiết 3 Vendor (SpeechAce, SpeechSuper, iFlytek) ở cấp độ PHONE LEVEL.
Sử dụng Canonical G2P (ARPA39) làm trục quy chiếu cố định.
Căn chỉnh Phone DP (Needleman-Wunsch) từ cả 3 vendor vào từng vị trí Canonical Phone.

Tính toán:
1. Ma trận tương quan Phone Score: Pearson r, Spearman rho, MAE, Bias giữa 3 cặp vendor.
2. Tỷ lệ đồng thuận 3-Way: 3/3 Đúng, 3/3 Sai, 2/3 Đa số, Bất đồng.
3. Độ tương đồng chẩn đoán âm thay thế (MSDD Substitution): SpeechAce sound_most_like vs SpeechSuper sound_like.
4. Độ lệch timestamp ranh giới âm vị: SpeechAce extent vs SpeechSuper span (ms).
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np

from g2p_en import G2p
from vh_gopt.core.phone_map import IPA2ARPA, _ifly_arpa

re_phone = re.compile(r"^([A-Z]+)[0-9]?$")
g2p = G2p()

# Bổ sung các biến thể IPA từ SpeechSuper
IPA2ARPA_FULL = dict(IPA2ARPA)
IPA2ARPA_FULL.update({
    "ɑ": "AA", "ɒ": "AA", "a": "AA", "æ": "AE", "ʌ": "AH", "ə": "AH",
    "ɔ": "AO", "aʊ": "AW", "aɪ": "AY", "b": "B", "tʃ": "CH", "d": "D",
    "ð": "DH", "ɛ": "EH", "e": "EY", "eɪ": "EY", "ɝ": "ER", "ɚ": "ER",
    "ɜ": "ER", "ər": "ER", "f": "F", "ɡ": "G", "g": "G", "h": "HH",
    "ɪ": "IH", "i": "IY", "iː": "IY", "dʒ": "JH", "k": "K", "l": "L",
    "m": "M", "n": "N", "ŋ": "NG", "oʊ": "OW", "o": "OW", "ɔɪ": "OY",
    "p": "P", "ɹ": "R", "r": "R", "s": "S", "ʃ": "SH", "t": "T",
    "ɾ": "T", "θ": "TH", "ʊ": "UH", "u": "UW", "uː": "UW", "v": "V",
    "w": "W", "j": "Y", "z": "Z", "ʒ": "ZH", "ɫ": "L"
})

def norm_word(w):
    return "".join(ch for ch in str(w).lower() if ch.isalnum())

def fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def pearson(x, y):
    if len(x) < 2: return float("nan")
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def spearman(x, y):
    if len(x) < 2: return float("nan")
    from scipy.stats import spearmanr
    res = spearmanr(x, y)
    return float(res.statistic if hasattr(res, "statistic") else res[0])

def align_sequence_dp(canon_phones, vendor_phones):
    """
    DP Needleman-Wunsch align vendor_phones [(arpa, score, diag_info)...]
    vao canon_phones [arpa1, arpa2, ...].
    Tra ve list dai bang canon_phones: moi vi tri la vendor item tuong ung hoac None.
    """
    n, m = len(canon_phones), len(vendor_phones)
    if n == 0:
        return []
    if m == 0:
        return [None] * n
    
    # Neu do dai va ky tu giong nhau 100% -> map 1-1 nhanh
    if n == m and all(canon_phones[i] == vendor_phones[i][0] for i in range(n)):
        return vendor_phones

    # DP matrix
    dp = np.zeros((n + 1, m + 1), dtype=float)
    for i in range(n + 1): dp[i, 0] = i * 1.0
    for j in range(m + 1): dp[0, j] = j * 1.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c_arpa = canon_phones[i - 1]
            v_arpa = vendor_phones[j - 1][0]
            cost = 0.0 if c_arpa == v_arpa else 1.0
            dp[i, j] = min(
                dp[i - 1, j - 1] + cost,      # match/sub
                dp[i - 1, j] + 1.0,          # del from canon
                dp[i, j - 1] + 1.0           # ins from vendor
            )
            
    # Backtrack
    i, j = n, m
    matched = [None] * n
    while i > 0 and j > 0:
        c_arpa = canon_phones[i - 1]
        v_arpa = vendor_phones[j - 1][0]
        cost = 0.0 if c_arpa == v_arpa else 1.0
        if dp[i, j] == dp[i - 1, j - 1] + cost:
            # Gán nếu cùng âm hoặc chấp nhận sub
            matched[i - 1] = vendor_phones[j - 1]
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + 1.0:
            matched[i - 1] = None
            i -= 1
        else:
            j -= 1
            
    return matched

def get_canonical_word_phones(target_text):
    """Tách target text thành các từ và canonical ARPA phones tương ứng cho từng từ."""
    raw_words = target_text.strip().split()
    word_tokens = []
    word_canon_phones = []
    
    for w in raw_words:
        nw = norm_word(w)
        if not nw:
            continue
        # G2P từng từ độc lập
        toks = g2p(nw)
        phones = []
        for t in toks:
            m = re_phone.match(t)
            if m:
                phones.append(m.group(1).upper())
        if phones:
            word_tokens.append(nw)
            word_canon_phones.append(phones)
            
    return word_tokens, word_canon_phones

def parse_speechace_phones(ace_data):
    """Trích xuất danh sách từ và phone của SpeechAce."""
    wsl = ace_data.get("word_score_list") or []
    words = []
    phones_per_word = []
    
    for w in wsl:
        nw = norm_word(w.get("word", ""))
        if not nw:
            continue
        w_phs = []
        for p in w.get("phone_score_list") or []:
            raw_p = (p.get("phone") or "").upper()
            m = re_phone.match(raw_p)
            arpa = m.group(1) if m else raw_p
            sc = float(p["quality_score"]) if fin(p.get("quality_score")) else None
            ext = p.get("extent") # [t_start, t_end]
            sub = p.get("sound_most_like")
            w_phs.append((arpa, sc, {"extent": ext, "sound_most_like": sub, "raw": p}))
        words.append(nw)
        phones_per_word.append(w_phs)
        
    return words, phones_per_word

def parse_speechsuper_phones(ss_data):
    """Trích xuất danh sách từ và phone của SpeechSuper."""
    res = ss_data.get("result") or {}
    wsl = res.get("words") or []
    words = []
    phones_per_word = []
    
    for w in wsl:
        nw = norm_word(w.get("word", ""))
        if not nw:
            continue
        w_phs = []
        for p in w.get("phonemes") or []:
            ipa = p.get("phoneme") or ""
            arpa = IPA2ARPA_FULL.get(ipa)
            if not arpa:
                # thử bỏ dấu
                clean_ipa = ipa.replace("ː", "").replace("ˑ", "")
                arpa = IPA2ARPA_FULL.get(clean_ipa, ipa.upper())
            sc = float(p["pronunciation"]) if fin(p.get("pronunciation")) else None
            span = p.get("span") # {'start': ms, 'end': ms}
            sub = p.get("sound_like")
            rtype = p.get("readType", 0)
            w_phs.append((arpa, sc, {"span": span, "sound_like": sub, "readType": rtype, "raw": p}))
        words.append(nw)
        phones_per_word.append(w_phs)
        
    return words, phones_per_word

def parse_iflytek_phones(ifly_data):
    """Trích xuất danh sách từ và phone của iFlytek."""
    words_raw = ifly_data.get("words") or []
    w_phones_raw = ifly_data.get("word_phones") or []
    w_diag_raw = ifly_data.get("word_diag") or []
    
    words = []
    phones_per_word = []
    
    for idx, w in enumerate(words_raw):
        nw = norm_word(w)
        if not nw:
            continue
        w_phs = []
        phs = w_phones_raw[idx] if idx < len(w_phones_raw) else []
        diags = w_diag_raw[idx] if idx < len(w_diag_raw) else []
        
        for k, p in enumerate(phs):
            sym = p[0] if isinstance(p, list) and len(p) > 0 else ""
            sc = float(p[1]) if isinstance(p, list) and len(p) > 1 and fin(p[1]) else None
            diag = diags[k] if k < len(diags) else {}
            lab = diag.get("label", "ok")
            
            # Tách cluster nếu có (ts, ar, dr...)
            arpa_list = _ifly_arpa(sym)
            for a in arpa_list:
                w_phs.append((a, sc, {"label": lab, "raw": diag}))
                
        words.append(nw)
        phones_per_word.append(w_phs)
        
    return words, phones_per_word

def align_words_to_canon(canon_words, vendor_words, vendor_phones_per_word):
    """Align các từ của vendor vào chuỗi từ canonical."""
    sm = SequenceMatcher(None, canon_words, vendor_words, autojunk=False)
    matched_vendor_phones = [[] for _ in range(len(canon_words))]
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                c_idx = i1 + k
                v_idx = j1 + k
                if v_idx < len(vendor_phones_per_word):
                    matched_vendor_phones[c_idx] = vendor_phones_per_word[v_idx]
                    
    return matched_vendor_phones

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/vh_alive_2026_scripted/manifest.clean9k.jsonl")
    ap.add_argument("--ace-dir", default="data/vh_alive_2026_scripted/vendor_eval")
    ap.add_argument("--ss-dir", default="cache/ss_vh")
    ap.add_argument("--ifly-dir", default="cache/iflytek_vh")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số mẫu (0 = tất cả)")
    ap.add_argument("--out-report", default="cache/vh_3vendor_phone_report.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
        
    print(f"=== ĐÁNH GIÁ 3 VENDOR PHONE-LEVEL TRÊN {len(rows)} MẪU SẠCH ===")
    
    total_canon_phones = 0
    aligned_phones = {
        "ace": 0, "ss": 0, "ifly": 0, "all3": 0
    }
    
    # Score pairs
    pairs_ace_ss = []
    pairs_ace_ifly = []
    pairs_ss_ifly = []
    
    # 3-Way agreements
    agr_3correct = 0
    agr_3error = 0
    agr_2majority = 0
    agr_disagree = 0
    
    # Boundary timing diffs (SpeechAce extent vs SpeechSuper span in ms)
    boundary_diffs_start = []
    boundary_diffs_end = []
    
    # MSDD substitution analysis
    msdd_sub_pairs = []
    
    for r_idx, row in enumerate(rows):
        ID = row["ID"]
        target = row.get("target_text", "")
        
        # 1. Canonical G2P
        canon_words, canon_word_phones = get_canonical_word_phones(target)
        if not canon_words:
            continue
            
        # 2. Load vendor files
        ace_p = Path(args.ace_dir) / f"{ID}.json"
        ss_p = Path(args.ss_dir) / f"{ID}.json"
        ifly_p = Path(args.ifly_dir) / f"{ID}.json"
        
        if not (ace_p.exists() and ss_p.exists() and ifly_p.exists()):
            continue
            
        try:
            ace_json = json.load(open(ace_p, encoding="utf-8"))["content"]["data"]
            ss_json = json.load(open(ss_p, encoding="utf-8"))
            ifly_json = json.load(open(ifly_p, encoding="utf-8"))
        except Exception:
            continue
            
        ace_words, ace_wphs = parse_speechace_phones(ace_json)
        ss_words, ss_wphs = parse_speechsuper_phones(ss_json)
        ifly_words, ifly_wphs = parse_iflytek_phones(ifly_json)
        
        # 3. Align vendor words to canonical words
        m_ace_wphs = align_words_to_canon(canon_words, ace_words, ace_wphs)
        m_ss_wphs = align_words_to_canon(canon_words, ss_words, ss_wphs)
        m_ifly_wphs = align_words_to_canon(canon_words, ifly_words, ifly_wphs)
        
        # 4. Align vendor phones to canonical phones within each word
        for w_idx in range(len(canon_words)):
            c_phs = canon_word_phones[w_idx]
            a_aligned = align_sequence_dp(c_phs, m_ace_wphs[w_idx])
            s_aligned = align_sequence_dp(c_phs, m_ss_wphs[w_idx])
            i_aligned = align_sequence_dp(c_phs, m_ifly_wphs[w_idx])
            
            for p_idx, c_p in enumerate(c_phs):
                total_canon_phones += 1
                
                a_item = a_aligned[p_idx] if p_idx < len(a_aligned) else None
                s_item = s_aligned[p_idx] if p_idx < len(s_aligned) else None
                i_item = i_aligned[p_idx] if p_idx < len(i_aligned) else None
                
                a_sc = a_item[1] if a_item and fin(a_item[1]) else None
                s_sc = s_item[1] if s_item and fin(s_item[1]) else None
                i_sc = i_item[1] if i_item and fin(i_item[1]) else None
                
                if a_sc is not None: aligned_phones["ace"] += 1
                if s_sc is not None: aligned_phones["ss"] += 1
                if i_sc is not None: aligned_phones["ifly"] += 1
                
                # Pair correlations
                if a_sc is not None and s_sc is not None:
                    pairs_ace_ss.append((a_sc, s_sc))
                if a_sc is not None and i_sc is not None:
                    pairs_ace_ifly.append((a_sc, i_sc))
                if s_sc is not None and i_sc is not None:
                    pairs_ss_ifly.append((s_sc, i_sc))
                    
                # 3-way check
                if a_sc is not None and s_sc is not None and i_sc is not None:
                    aligned_phones["all3"] += 1
                    
                    # Ngưỡng đúng/sai: >= 60 là đạt
                    a_ok = a_sc >= 60.0
                    s_ok = s_sc >= 60.0
                    i_ok = i_sc >= 60.0
                    votes = [a_ok, s_ok, i_ok]
                    
                    if votes.count(True) == 3:
                        agr_3correct += 1
                    elif votes.count(False) == 3:
                        agr_3error += 1
                    elif votes.count(True) == 2 or votes.count(False) == 2:
                        agr_2majority += 1
                    else:
                        agr_disagree += 1
                        
                    # Boundary timestamp cross-check (SpeechAce extent vs SpeechSuper span)
                    a_ext = a_item[2].get("extent") # centiseconds in SpeechAce
                    s_span = s_item[2].get("span")   # centiseconds in SpeechSuper
                    dur_ms = row.get("duration_sec", 0.0) * 1000.0
                    if a_ext and s_span and isinstance(a_ext, list) and len(a_ext) == 2 and isinstance(s_span, dict):
                        # Cả SpeechAce và SpeechSuper đều dùng đơn vị centisecond (10ms frame) -> đổi ra ms bằng * 10
                        a_st, a_en = a_ext[0] * 10.0, a_ext[1] * 10.0
                        s_st, s_en = float(s_span.get("start", 0)) * 10.0, float(s_span.get("end", 0)) * 10.0
                        # Sanity filter: loại bỏ rác bộ nhớ uninitialized int (vd 1222419744 từ SS)
                        if 0 <= a_st <= dur_ms and 0 <= s_st <= dur_ms and 0 <= a_en <= dur_ms + 500 and 0 <= s_en <= dur_ms + 500:
                            boundary_diffs_start.append(abs(a_st - s_st))
                            boundary_diffs_end.append(abs(a_en - s_en))
                            
                    # MSDD substitution cross-check (SpeechAce ARPABET vs SpeechSuper IPA -> ARPA39)
                    a_sub = a_item[2].get("sound_most_like")
                    s_sub_ipa = s_item[2].get("sound_like")
                    if a_sub and s_sub_ipa:
                        a_arpa = "".join(c for c in a_sub.upper() if c.isalpha())
                        s_arpa = IPA2ARPA_FULL.get(s_sub_ipa, s_sub_ipa.upper())
                        s_arpa = "".join(c for c in s_arpa if c.isalpha())
                        if a_arpa and s_arpa:
                            is_err = (a_sc < 60.0 or s_sc < 60.0)
                            ace_is_sub = (a_arpa != c_p)
                            ss_is_sub = (s_arpa != c_p)
                            msdd_sub_pairs.append({
                                "orig": c_p, "ace": a_arpa, "ss": s_arpa,
                                "is_err": is_err, "ace_sub": ace_is_sub, "ss_sub": ss_is_sub,
                                "mutual_sub": (is_err and ace_is_sub and ss_is_sub),
                                "exact_match": (is_err and ace_is_sub and ss_is_sub and a_arpa == s_arpa)
                            })
    def stats(pairs, name):
        if not pairs:
            return {}
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mae = float(np.mean(np.abs(np.array(xs) - np.array(ys))))
        bias = float(np.mean(np.array(xs) - np.array(ys)))
        r = pearson(xs, ys)
        rho = spearman(xs, ys)
        return {
            "n": len(pairs),
            "pearson": round(r, 4),
            "spearman": round(rho, 4),
            "mae": round(mae, 2),
            "bias(A-B)": round(bias, 2)
        }

    report = {
        "total_canonical_phones": total_canon_phones,
        "aligned_phones": aligned_phones,
        "phone_coverage_pct": {
            k: round(100 * v / max(1, total_canon_phones), 2) for k, v in aligned_phones.items()
        },
        "correlations": {
            "SpeechAce_vs_SpeechSuper": stats(pairs_ace_ss, "ACE-SS"),
            "SpeechAce_vs_iFlytek": stats(pairs_ace_ifly, "ACE-IFLY"),
            "SpeechSuper_vs_iFlytek": stats(pairs_ss_ifly, "SS-IFLY")
        },
        "three_way_agreement": {
            "total_all3_evaluated": aligned_phones["all3"],
            "all_3_correct": agr_3correct,
            "all_3_correct_pct": round(100 * agr_3correct / max(1, aligned_phones["all3"]), 2),
            "all_3_error": agr_3error,
            "all_3_error_pct": round(100 * agr_3error / max(1, aligned_phones["all3"]), 2),
            "majority_2_of_3": agr_2majority,
            "majority_2_of_3_pct": round(100 * agr_2majority / max(1, aligned_phones["all3"]), 2),
            "consensus_usable_pct (3/3 + 2/3)": round(100 * (agr_3correct + agr_3error + agr_2majority) / max(1, aligned_phones["all3"]), 2)
        },
        "boundary_timing_mae_ms": {
            "count": len(boundary_diffs_start),
            "start_median": round(float(np.median(boundary_diffs_start)), 1) if boundary_diffs_start else None,
            "start_mean": round(float(np.mean(boundary_diffs_start)), 1) if boundary_diffs_start else None,
            "start_p75": round(float(np.percentile(boundary_diffs_start, 75)), 1) if boundary_diffs_start else None,
            "start_p90": round(float(np.percentile(boundary_diffs_start, 90)), 1) if boundary_diffs_start else None,
            "end_median": round(float(np.median(boundary_diffs_end)), 1) if boundary_diffs_end else None,
            "end_mean": round(float(np.mean(boundary_diffs_end)), 1) if boundary_diffs_end else None,
            "end_p75": round(float(np.percentile(boundary_diffs_end, 75)), 1) if boundary_diffs_end else None,
            "end_p90": round(float(np.percentile(boundary_diffs_end, 90)), 1) if boundary_diffs_end else None
        },
        "msdd_true_substitution_diagnosis": {
            "total_pairs_evaluated": len(msdd_sub_pairs),
            "error_flagged_phones": sum(1 for p in msdd_sub_pairs if p["is_err"]),
            "mutual_substitution_phones": sum(1 for p in msdd_sub_pairs if p["mutual_sub"]),
            "exact_same_substituted_phone": sum(1 for p in msdd_sub_pairs if p["exact_match"]),
            "exact_agreement_pct_on_mutual_sub": round(
                100 * sum(1 for p in msdd_sub_pairs if p["exact_match"]) / max(1, sum(1 for p in msdd_sub_pairs if p["mutual_sub"])), 2
            )
        }
    }

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ 3 VENDOR PHONE-LEVEL")
    print("=" * 60)
    print(f"Tổng Canonical Phones  : {total_canon_phones:,}")
    print(f"Phones đủ cả 3 Vendor  : {aligned_phones['all3']:,} ({report['phone_coverage_pct']['all3']}%)")
    print("\n--- MA TRẬN TƯƠNG QUAN PHONE SCORE ---")
    for pair_name, st in report["correlations"].items():
        print(f"  {pair_name:25}: N={st['n']:,} | r={st['pearson']:.4f} | rho={st['spearman']:.4f} | MAE={st['mae']} | Bias={st['bias(A-B)']}")
    
    print("\n--- ĐỘ ĐỒNG THUẬN 3-WAY VOTING ---")
    twa = report["three_way_agreement"]
    print(f"  Cả 3 Vendor cùng ĐÚNG (3/3) : {twa['all_3_correct']:,} ({twa['all_3_correct_pct']}%)")
    print(f"  Cả 3 Vendor cùng SAI  (3/3) : {twa['all_3_error']:,} ({twa['all_3_error_pct']}%)")
    print(f"  Đa số đồng thuận      (2/3) : {twa['majority_2_of_3']:,} ({twa['majority_2_of_3_pct']}%)")
    print(f"  ==> TỔNG DÙNG ĐƯỢC CHO LABEL: {twa['consensus_usable_pct (3/3 + 2/3)']}%")
    
    if report["boundary_timing_mae_ms"]["start_median"] is not None:
        bt = report["boundary_timing_mae_ms"]
        print(f"\n--- RANH GIỚI THỜI GIAN (ACE vs SS trên {bt['count']:,} âm vị) ---")
        print(f"  Start Diff: Median={bt['start_median']} ms | P75={bt['start_p75']} ms | P90={bt['start_p90']} ms | Mean={bt['start_mean']} ms")
    ms = report["msdd_true_substitution_diagnosis"]
    print(f"\n--- CHẨN ĐOÁN LỖI THAY THẾ MSDD THẬT (LOẠI BỎ ÂM ĐÚNG) ---")
    print(f"  Số âm vị bị flag lỗi (score < 60)         : {ms['error_flagged_phones']:,}")
    print(f"  Cả 2 cùng xác nhận bị đổi âm (Mutual Sub) : {ms['mutual_substitution_phones']:,}")
    print(f"  Đồng thuận CHÍNH XÁC cùng 1 âm thay thế   : {ms['exact_same_substituted_phone']:,} ({ms['exact_agreement_pct_on_mutual_sub']}%)")
    print(f"\nĐã ghi báo cáo chi tiết vào: {args.out_report}")

if __name__ == "__main__":
    main()
