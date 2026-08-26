"""
Path-A (GOP-79d): dùng KoelLabs/xlsr-english-01 làm acoustic model TRONG chính GOP.
Map âm chuẩn ARPA39 -> token IPA -> id trong vocab KoelLabs (80 lớp gồm <pad>=blank).

Vocab KoelLabs chứa đủ 39 âm ARPA dưới dạng token ĐƠN (kể cả diphthong/affricate:
aɪ aʊ oʊ eɪ ɔɪ tʃ dʒ ŋ), nên map 1-1 sạch. Dùng chung cho gopt_data/koel_extract_gop
(trích feature) và gopt_infer/eval_kid (inference mô hình GOP-79d).
"""
import sys

# ARPA39 -> IPA khớp token trong vocab KoelLabs (R->ɹ, G->ɡ script-g U+0261)
ARPA2KOEL_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b',
    'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f',
    'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k', 'L': 'l',
    'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ',
    'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}


def build_arpa2id(tokenizer):
    """-> dict ARPA -> id (int) trong vocab KoelLabs. Lỗi rõ nếu thiếu token."""
    vocab = tokenizer.get_vocab()
    a2id, missing = {}, []
    for arpa, ipa in ARPA2KOEL_IPA.items():
        if ipa in vocab:
            a2id[arpa] = vocab[ipa]
        else:
            missing.append((arpa, ipa))
    if missing:
        raise ValueError(f"ARPA thiếu token IPA trong vocab KoelLabs: {missing}\nvocab={sorted(vocab)}")
    return a2id


def map_phones_to_ids_koel(phones, tokenizer, _cache={}):
    """Giống infer_gop.map_phones_to_ids nhưng ARPA -> IPA -> id KoelLabs."""
    key = id(tokenizer)
    a2id = _cache.get(key)
    if a2id is None:
        a2id = _cache[key] = build_arpa2id(tokenizer)
    ids, mapped = [], []
    for p in phones:
        base = "".join(c for c in p if c.isalpha())  # bỏ trọng âm nếu còn
        if base not in a2id:
            sys.exit(f"Phone ARPA '{p}' không map được sang vocab KoelLabs")
        ids.append(a2id[base]); mapped.append(base)
    return ids, mapped


if __name__ == "__main__":                       # verify: 39/39 map được
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained("KoelLabs/xlsr-english-01").tokenizer
    a2id = build_arpa2id(tok)
    print(f"OK: {len(a2id)}/39 âm ARPA map sang id KoelLabs")
    inv = {i: t for t, i in tok.get_vocab().items()}
    for a in sorted(a2id):
        print(f"  {a:3} -> {ARPA2KOEL_IPA[a]:3} (id {a2id[a]}, tok {inv[a2id[a]]!r})")
