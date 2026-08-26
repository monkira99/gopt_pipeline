"""
Fast GOP feature-vector extraction (is24 gop-af-feats style), vectorized.

Key speedup: the P substitution candidates at a phone position share the same
sequence length, differing only at that position -> run ONE batched CTC forward
(batch dim = candidates) instead of P separate Python-loop forwards.
"""
import torch


def ctc_forward_batch(params, seqmat, blank=0):
    """
    params : [P, T]  (shared emission posteriors, float64)
    seqmat : [B, S]  (int64 token ids; B candidate sequences of equal length S)
    returns: [B]     forward probability P(seq_b) for each candidate (NOT log)
    """
    P, T = params.shape
    B, S = seqmat.shape
    L = 2 * S + 1
    alphas = torch.zeros((B, L, T), dtype=torch.float64)
    bar = torch.arange(B)

    alphas[:, 0, 0] = params[blank, 0]
    alphas[:, 1, 0] = params[seqmat[:, 0], 0]

    for t in range(1, T):
        start = max(0, L - 2 * (T - t))
        for s in range(start, L):
            l = (s - 1) // 2
            if s % 2 == 0:  # blank state
                if s == 0:
                    alphas[:, s, t] = alphas[:, s, t - 1] * params[blank, t]
                else:
                    alphas[:, s, t] = (alphas[:, s, t - 1] + alphas[:, s - 1, t - 1]) * params[blank, t]
            else:
                tok = seqmat[:, l]                      # [B]
                emit = params[tok, t]                   # [B]
                two = alphas[:, s, t - 1] + alphas[:, s - 1, t - 1]
                if s == 1:
                    alphas[:, s, t] = two * emit
                else:
                    same = seqmat[:, l] == seqmat[:, l - 1]     # [B] bool
                    three = two + alphas[:, s - 2, t - 1]
                    base = torch.where(same, two, three)
                    alphas[:, s, t] = base * emit
    return alphas[:, L - 1, T - 1] + alphas[:, L - 2, T - 1]


def ctc_forward_single(params, seq, blank=0):
    """Non-batched forward prob for one sequence (used for deletion, variable length)."""
    return ctc_forward_batch(params, seq.view(1, -1), blank=blank)[0]


def ctc_forward_batch_norm(params, seqmat, blank=0):
    """
    taslpro26 (2026) NORMALIZED scaled-forward, batched over candidates.
    params : [P, T] ; seqmat : [B, S] -> returns NLL [B] (= -sum_t log alpha_bar).
    """
    P, T = params.shape
    B, S = seqmat.shape
    L = 2 * S + 1
    alphas = torch.zeros((B, L, T), dtype=torch.float64)
    alpha_bar = torch.zeros((B, T), dtype=torch.float64)

    alphas[:, 0, 0] = params[blank, 0]
    alphas[:, 1, 0] = params[seqmat[:, 0], 0]
    alpha_bar[:, 0] = alphas[:, :, 0].sum(1)
    alphas[:, :, 0] = alphas[:, :, 0] / alpha_bar[:, 0:1]

    for t in range(1, T):
        start = max(0, L - 2 * (T - t))
        for s in range(start, L):
            l = (s - 1) // 2
            if s % 2 == 0:
                if s == 0:
                    alphas[:, s, t] = alphas[:, s, t - 1] * params[blank, t]
                else:
                    alphas[:, s, t] = (alphas[:, s, t - 1] + alphas[:, s - 1, t - 1]) * params[blank, t]
            else:
                tok = seqmat[:, l]
                emit = params[tok, t]
                two = alphas[:, s, t - 1] + alphas[:, s - 1, t - 1]
                if s == 1:
                    alphas[:, s, t] = two * emit
                else:
                    same = seqmat[:, l] == seqmat[:, l - 1]
                    three = two + alphas[:, s - 2, t - 1]
                    alphas[:, s, t] = torch.where(same, two, three) * emit
        alpha_bar[:, t] = alphas[:, :, t].sum(1)
        alpha_bar[:, t] = torch.clamp_min(alpha_bar[:, t], 1e-300)
        alphas[:, :, t] = alphas[:, :, t] / alpha_bar[:, t:t + 1]
    return -(torch.log(alpha_bar).sum(1))


def extract_utt_feats_norm(params, labels, blank=0):
    """
    2026 GOP-SF feature vector: [LPP(=NLL_canon)] + [LPR per token] using the
    NORMALIZED forward. LPR = -NLL_canon + NLL_candidate. id 0 slot = deletion.
    """
    P, T = params.shape
    S = labels.shape[0]
    labels = labels.long()
    nll_canon = ctc_forward_batch_norm(params, labels.view(1, -1), blank=blank)[0]
    feats = torch.zeros((S, 1 + P), dtype=torch.float64)
    feats[:, 0] = nll_canon
    for i in range(S):
        seqmat = labels.view(1, -1).repeat(P, 1)
        seqmat[:, i] = torch.arange(P)
        nll = ctc_forward_batch_norm(params, seqmat, blank=blank)   # [P]
        lpr = -nll_canon + nll
        if S > 1:
            del_seq = torch.cat([labels[:i], labels[i + 1:]])
            nll_del = ctc_forward_batch_norm(params, del_seq.view(1, -1), blank=blank)[0]
            lpr[blank] = -nll_canon + nll_del
        feats[i, 1:] = lpr
    return feats, nll_canon.item()


def canonical_occupancy(params, labels, blank=0):
    """
    Per-phone occupancy = expected #frames the phone state is occupied, via a
    log-space CTC forward-backward on the CANONICAL sequence (one pass/utt, cheap).
    Returns occ [S] (soft duration; sums roughly to T over all phones+blanks).
    """
    P, T = params.shape
    S = labels.shape[0]
    labels = labels.long()
    L = 2 * S + 1
    logp = torch.log(params.clamp_min(1e-300))            # [P,T]
    NEG = -1e30

    def tok(s):                                            # token id at CTC state s
        return blank if s % 2 == 0 else int(labels[(s - 1) // 2])

    la = torch.full((L, T), NEG, dtype=torch.float64)
    lb = torch.full((L, T), NEG, dtype=torch.float64)
    la[0, 0] = logp[blank, 0]
    la[1, 0] = logp[labels[0], 0]
    for t in range(1, T):
        for s in range(L):
            prev = [la[s, t - 1]]
            if s - 1 >= 0:
                prev.append(la[s - 1, t - 1])
            if s % 2 == 1 and s - 2 >= 0 and labels[(s - 1) // 2] != labels[(s - 3) // 2]:
                prev.append(la[s - 2, t - 1])
            la[s, t] = torch.logsumexp(torch.stack(prev), 0) + logp[tok(s), t]

    lb[L - 1, T - 1] = 0.0
    lb[L - 2, T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        for s in range(L):
            nxt = [lb[s, t + 1] + logp[tok(s), t + 1]]
            if s + 1 < L:
                nxt.append(lb[s + 1, t + 1] + logp[tok(s + 1), t + 1])
            if s % 2 == 1 and s + 2 < L and labels[(s - 1) // 2] != labels[(s + 1) // 2]:
                nxt.append(lb[s + 2, t + 1] + logp[tok(s + 2), t + 1])
            lb[s, t] = torch.logsumexp(torch.stack(nxt), 0)

    logZ = torch.logsumexp(torch.stack([la[L - 1, T - 1], la[L - 2, T - 1]]), 0)
    gamma = torch.exp(la + lb - logZ)                     # [L,T] state posteriors
    occ = torch.zeros(S, dtype=torch.float64)
    for i in range(S):
        occ[i] = gamma[2 * i + 1].sum()
    return occ


def extract_utt_feats_norm_fast(params, labels, blank=0, occ=True, cap_elems=4e7):
    """
    Same 41-d GOP-SF feature as extract_utt_feats_norm, but batches all phone
    positions into few forward passes (chunked to respect `cap_elems` alpha
    tensor size). Returns (feats [S,1+P], occ [S] or None).
    """
    P, T = params.shape
    S = labels.shape[0]
    labels = labels.long()
    nll_canon = ctc_forward_batch_norm(params, labels.view(1, -1), blank=blank)[0]

    feats = torch.zeros((S, 1 + P), dtype=torch.float64)
    feats[:, 0] = nll_canon

    per_pos = P * (2 * S + 1) * T
    chunk = max(1, int(cap_elems // max(per_pos, 1)))
    for a in range(0, S, chunk):
        b = min(S, a + chunk)
        idx = list(range(a, b))
        # substitutions: for each pos i, labels with col i replaced by arange(P)
        seqmat = labels.view(1, -1).repeat(len(idx) * P, 1)
        for k, i in enumerate(idx):
            seqmat[k * P:(k + 1) * P, i] = torch.arange(P)
        nll = ctc_forward_batch_norm(params, seqmat, blank=blank).view(len(idx), P)
        feats[a:b, 1:] = -nll_canon + nll
    # deletions (all length S-1) in one batch
    if S > 1:
        del_seqs = torch.stack([torch.cat([labels[:i], labels[i + 1:]]) for i in range(S)])
        nll_del = ctc_forward_batch_norm(params, del_seqs, blank=blank)
        feats[:, 1 + blank] = -nll_canon + nll_del

    occ_out = canonical_occupancy(params, labels, blank=blank) if occ else None
    return feats, occ_out


def extract_utt_feats(params, labels, blank=0):
    """
    params : [P, T] float64 ; labels : [S] int (canonical ids)
    returns: feats [S, 1+P]  (col 0 = LPP; col 1..P = LPR when replacing phone i
             by token id (id 0 == deletion)); and lpp scalar.
    LPR = logP(canonical) - logP(candidate).
    """
    P, T = params.shape
    S = labels.shape[0]
    labels = labels.long()
    p_self = ctc_forward_single(params, labels, blank=blank)     # scalar prob
    lpp = torch.log(p_self)                                       # logP(canonical) (<=0)

    feats = torch.zeros((S, 1 + P), dtype=torch.float64)
    feats[:, 0] = lpp
    for i in range(S):
        # substitutions: batch over all token ids at position i (includes blank slot, unused)
        seqmat = labels.view(1, -1).repeat(P, 1)
        seqmat[:, i] = torch.arange(P)
        probs = ctc_forward_batch(params, seqmat, blank=blank)    # [P]
        logp = torch.log(probs.clamp_min(1e-300))
        lpr = lpp - logp                                          # [P]
        # deletion for the id==blank slot
        if S > 1:
            del_seq = torch.cat([labels[:i], labels[i + 1:]])
            pdel = ctc_forward_single(params, del_seq, blank=blank)
            lpr[blank] = lpp - torch.log(pdel.clamp_min(1e-300))
        feats[i, 1:] = lpr
    return feats, lpp.item()
