#!/usr/bin/env python3
"""
Self-contained CTC-based GOP inference for a SINGLE (audio, reference-text) pair.

Faithful re-implementation of the GOP math from frank613/CTC-based-GOP:
  - ctc_loss()        : CTC forward log-likelihood of the canonical phone sequence
  - ctc_loss_denom()  : segmentation-free denominator with an "arbitrary" (wildcard)
                        state at one position + skip paths  (GOP-SF / SD variant)
  scalar GOP(i) = -ll_self + ll_denom(i)   (== log P(canonical) - log P(any-substitution))

Adds a G2P front-end so you can pass plain English text instead of a CTM file.

Usage:
  python infer_gop.py --audio sample.wav --text "she had your dark suit" \
      --model ./model --processor ./model

  # or give phones directly (space-separated ARPAbet, no stress):
  python infer_gop.py --audio sample.wav --phones "SH IY HH AE D" --model ./model
"""
import argparse
import re
import sys
import torch

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')

ARPA39 = {'AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G',
          'HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T',
          'TH','UH','UW','V','W','Y','Z','ZH'}


# ----------------------------- GOP core (verbatim math) -----------------------------
def ctc_loss(params, seq, blank=0):
    """CTC forward. params: [P tokens, T frames]. Returns -logLik (NLL)."""
    seqLen = seq.shape[0]
    L = 2 * seqLen + 1
    T = params.shape[1]
    alphas = torch.zeros((L, T)).double()
    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    for t in range(1, T):
        start = max(0, L - 2 * (T - t))
        for s in range(start, L):
            l = int((s - 1) / 2)
            if s % 2 == 0:  # blank
                if s == 0:
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
            elif s == 1 or seq[l] == seq[l - 1]:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
            else:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                    * params[seq[l], t]
    llForward = torch.log(alphas[L - 1, T - 1] + alphas[L - 2, T - 1])
    return -llForward


def check_arbitrary(in_alphas, s, t, zero_pos=[]):
    if in_alphas[s, t].sum() > 0:
        if len(zero_pos) != 0:
            mask = torch.ones_like(in_alphas[s, t])
            for i in zero_pos:
                mask[i] = 0
            return sum(in_alphas[s, t][mask.bool()])
        return sum(in_alphas[s, t][:])
    return False


def ctc_loss_denom(params, seq, pos, blank=0):
    """Segmentation-free denominator with a wildcard state at `pos`. Returns -logLik."""
    seqLen = seq.shape[0]
    L = 2 * seqLen + 1
    T = params.shape[1]
    P = params.shape[0]
    mask_ins = torch.eye(P)
    alphas = torch.zeros((L, T, P)).double()
    if pos == 0:
        alphas[0, 0, 0] = params[blank, 0]
        alphas[2, 0, 0] = 0
        alphas[3, 0, 0] = params[seq[1], 0]
        alphas[1, 0] = params[0:, 0]
        alphas[1, 0, 0] = 0
    else:
        alphas[0, 0, 0] = params[blank, 0]
        alphas[1, 0, 0] = params[seq[0], 0]
    for t in range(1, T):
        if pos == seqLen - 1:
            lowest_state = L - 2 * (T - t + 1)
        else:
            lowest_state = L - 2 * (T - t)
        start = max(0, lowest_state)
        for s in range(start, L):
            l = int((s - 1) / 2)
            if s % 2 == 0:  # blank
                if s == 0:
                    alphas[s, t, 0] = alphas[s, t - 1, 0] * params[blank, t]
                else:
                    ssum = check_arbitrary(alphas, s - 1, t - 1, [blank])
                    if ssum:
                        alphas[s, t, 0] = (alphas[s, t - 1, 0] + ssum) * params[blank, t]
                    else:
                        alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]) * params[blank, t]
            elif pos != l and pos != l - 1:
                if s == 1 or seq[l] == seq[l - 1]:
                    alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]) * params[seq[l], t]
                else:
                    alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + alphas[s - 2, t - 1, 0]) \
                        * params[seq[l], t]
            elif pos == l - 1:
                ssum = check_arbitrary(alphas, s - 2, t - 1, [blank, seq[l]])
                if l - 2 < 0 or seq[l - 2] == seq[l]:
                    skip_token = 0
                else:
                    skip_token = alphas[s - 4, t - 1, 0] * params[seq[l], t]
                skip_empty = alphas[s - 3, t - 1, 0] * params[seq[l], t]
                alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + ssum) * params[seq[l], t] \
                    + skip_empty + skip_token
            else:  # wildcard state
                if s == 1:
                    empty_prob = alphas[s - 1, t - 1, 0] * params[:, t]
                    empty_prob[0] = 0
                    alphas[s, t, :] = (alphas[s, t - 1, :].view(1, -1) * params[:, t].view(-1, 1) * mask_ins).sum(-1) + empty_prob
                else:
                    skip_prob = alphas[s - 2, t - 1, 0] * params[:, t]
                    skip_prob[seq[l - 1]] = 0
                    skip_prob[0] = 0
                    empty_prob = alphas[s - 1, t - 1, 0] * params[:, t]
                    empty_prob[0] = 0
                    alphas[s, t, :] = (alphas[s, t - 1, :].view(1, -1) * params[:, t].view(-1, 1) * mask_ins).sum(-1) + skip_prob + empty_prob
    ssum = check_arbitrary(alphas, L - 2, T - 1, [blank])
    if ssum:
        llForward = torch.log(alphas[L - 1, T - 1, 0] + ssum + alphas[L - 3, T - 1, 0] + alphas[L - 4, T - 1, 0])
    else:
        llForward = torch.log(alphas[L - 1, T - 1, 0] + alphas[L - 2, T - 1, 0])
    return -llForward


# ----------------------------- front-end helpers -----------------------------
def text_to_arpa(text):
    """English text -> list of ARPAbet-39 phones (stress stripped) using g2p_en."""
    try:
        from g2p_en import G2p
    except ImportError:
        sys.exit("g2p_en not installed. Run: uv pip install g2p_en  (or pass --phones directly)")
    g2p = G2p()
    phones = []
    for tok in g2p(text):
        m = re_phone.match(tok)
        if not m:
            continue
        p = m.group(1).upper()
        if p in ARPA39:
            phones.append(p)
    return phones


def map_phones_to_ids(phones, tokenizer):
    """Map ARPA phones to this model's vocab ids, tolerant of upper/lower case."""
    vocab = tokenizer.get_vocab()
    lower = {k.lower(): v for k, v in vocab.items()}
    ids, mapped = [], []
    for p in phones:
        key = p if p in vocab else (p.lower() if p.lower() in lower else None)
        if key is None:
            sys.exit(f"Phone '{p}' not in model vocab. Model vocab: {sorted(vocab)}")
        ids.append(vocab.get(p, lower.get(p.lower())))
        mapped.append(p)
    return ids, mapped


def detect_blank_id(tokenizer, model):
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return pad
    v = tokenizer.get_vocab()
    for k in ("<pad>", "[PAD]", "<blank>"):
        if k in v:
            return v[k]
    return getattr(model.config, "pad_token_id", 0) or 0


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", default=None, help="reference text (English)")
    ap.add_argument("--phones", default=None, help="space-separated ARPAbet phones (alt to --text)")
    ap.add_argument("--model", default="./model")
    ap.add_argument("--processor", default=None, help="processor dir (default: same as --model)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                    help="model forward device; GOP DP always runs on CPU/double")
    ap.add_argument("--feats", action="store_true", help="also output the per-phone GOP feature VECTOR")
    args = ap.parse_args()
    if not args.text and not args.phones:
        ap.error("provide --text or --phones")

    import librosa
    from transformers.models.wav2vec2 import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

    proc_dir = args.processor or args.model
    processor = Wav2Vec2Processor.from_pretrained(proc_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(proc_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).eval().to(args.device)
    blank = detect_blank_id(tokenizer, model)

    phones = args.phones.split() if args.phones else text_to_arpa(args.text)
    if not phones:
        sys.exit("No phones produced from input.")
    ids, mapped = map_phones_to_ids(phones, tokenizer)
    labels = torch.tensor(ids, dtype=torch.int32)

    speech, _ = librosa.load(args.audio, sr=16000, mono=True)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(args.device)

    with torch.no_grad():
        logits = model(input_values).logits.squeeze(0)
    post_mat = logits.softmax(dim=-1).cpu().double()          # [T, P] (MPS has no f64 -> cpu first)
    params = post_mat.transpose(0, 1)                          # [P, T]

    ll_self = ctc_loss(params, labels, blank=blank)
    print(f"# audio={args.audio}")
    print(f"# phones({len(mapped)}): {' '.join(mapped)}")
    print(f"# blank_id={blank}  frames={params.shape[1]}  vocab={params.shape[0]}")
    print(f"# canonical CTC logLik = {(-ll_self).item():.3f}")
    print("idx phone   GOP")
    gops = []
    for i, pid in enumerate(labels.tolist()):
        ll_denom = ctc_loss_denom(params, labels, i, blank=blank)
        gop = (-ll_self + ll_denom).item()
        gops.append(gop)
        print(f"{i:3d} {mapped[i]:5s} {gop:9.3f}")
    print(f"# mean GOP = {sum(gops)/len(gops):.3f}   (higher = better; ~0 best, more negative = worse)")

    if args.feats:
        num_token = params.shape[0]
        print("\n# --- GOP feature vectors (LPP, then LPR per token id) ---")
        for i, pid in enumerate(labels.tolist()):
            feats = [(-ll_self).item()]
            for sub in range(num_token):
                nl = labels.clone()
                if sub == blank:
                    nl = torch.cat([nl[:i], nl[i + 1:]])
                else:
                    nl[i] = sub
                fp = torch.exp(-ctc_loss(params, nl, blank=blank))
                feats.append(((-ll_self) - torch.log(fp)).item() if fp > 0 else float('nan'))
            print(f"{i} {mapped[i]} " + ",".join(f"{x:.3f}" for x in feats))


if __name__ == "__main__":
    main()
