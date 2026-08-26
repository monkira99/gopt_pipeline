#!/usr/bin/env python3
"""
Map phone của 2 vendor về ARPA39 canonical (g2p) để lấy PHONE CONSENSUS.

- SpeechSuper: dùng IPA -> ARPA (nghịch đảo ARPA2KOEL_IPA + biến thể SS: ə,e,r).
- iFlytek: dùng ARPABET viết thường -> ARPA; 6 cluster/rhotic tách thành 2 âm
  (ts,ar,ir,dr,ur) hoặc 1 (oo,ax) để KHỚP ĐỘ DÀI với chuỗi canonical.

Tín hiệu phone của cả hai là LABEL ok/sub/del (SS: readType; iFlytek: dp_message) ->
consensus nhị phân: cả 2 "đúng"->2.0, cả 2 "sai"->0.0, bất đồng hoặc lệch-align -> mask (None).
"""
from vh_gopt.core.koel_gop import ARPA2KOEL_IPA

# SS IPA -> ARPA (nghịch đảo + biến thể SS thực tế thấy trong cache)
IPA2ARPA = {v: k for k, v in ARPA2KOEL_IPA.items()}
IPA2ARPA.update({"ə": "AH", "e": "EY", "r": "R", "ɜ": "ER", "ɚ": "ER", "ɾ": "T", "ɫ": "L"})

# iFlytek lowercase -> ARPA. Giá trị là LIST (cluster tách nhiều âm) để khớp độ dài canonical.
IFLY2ARPA = {
    "ts": ["T", "S"], "ar": ["AA", "R"], "ir": ["IH", "R"],
    "dr": ["D", "R"], "ur": ["UH", "R"], "oo": ["UH"], "ax": ["AH"],
}


def _ifly_arpa(sym):
    s = sym.lower()
    if s in IFLY2ARPA:
        return IFLY2ARPA[s]
    return [s.upper()]


def ok_labels(diag_word, backend):
    """word_diag của 1 từ -> list ok_bool ĐÃ khớp-độ-dài-canonical (cluster iFlytek nhân đôi label).
    backend: 'ss' | 'iflytek'. Trả None nếu rỗng."""
    if not diag_word:
        return None
    out = []
    for d in diag_word:
        ok = d.get("label") == "ok"
        if backend == "iflytek":
            n = len(_ifly_arpa(d.get("phoneme", "")))
            out += [ok] * n            # cluster -> gán cùng label cho các âm tách ra
        else:
            out.append(ok)
    return out


def ss_scores_arpa(ss_word_phones):
    """SS word_phones [[ipa, score0-100]...] -> list score (theo thứ tự, đã map ipa->arpa 1-1)."""
    if not ss_word_phones:
        return None
    return [None if (s is None or (isinstance(s, float) and s != s)) else float(s)
            for _, s in ss_word_phones]


def phone_graded(canon_phones, ss_word_phones, ifl_diag_word, thr=50.0):
    """Nhãn phone 0-2 dạng GRADED: điểm SS (0-100/50) tại phone mà iFlytek ALIGN & CÙNG CHIỀU
    đúng/sai (SS>=thr <-> iFlytek label ok). Lệch align / bất đồng chiều -> None (mask).
    canon_phones: ARPA g2p của 1 từ."""
    n = len(canon_phones)
    ss = ss_scores_arpa(ss_word_phones)
    ifl = ok_labels(ifl_diag_word, "iflytek")
    if ss is None or ifl is None or len(ss) != n or len(ifl) != n:
        return [None] * n
    out = []
    for sc, ifl_ok in zip(ss, ifl):
        if sc is None:
            out.append(None); continue
        ss_ok = sc >= thr
        out.append(min(max(sc / 50.0, 0.0), 2.0) if ss_ok == ifl_ok else None)  # cùng chiều -> giữ điểm SS
    return out


def phone_consensus(canon_phones, ss_diag_word, ifl_diag_word):
    """canon_phones: list ARPA (g2p) của 1 từ. -> list nhãn phone 0-2 hoặc None (mask) mỗi vị trí.
    Chỉ gán khi CẢ HAI vendor align đúng độ dài canonical VÀ đồng thuận ok/not-ok."""
    n = len(canon_phones)
    ss = ok_labels(ss_diag_word, "ss")
    ifl = ok_labels(ifl_diag_word, "iflytek")
    if ss is None or ifl is None or len(ss) != n or len(ifl) != n:
        return [None] * n                  # lệch align -> mask cả từ (agree-or-drop)
    out = []
    for a, b in zip(ss, ifl):
        out.append(2.0 if (a and b) else (0.0 if (not a and not b) else None))
    return out
