"""Fast GOP feature-vector extraction (is24 gop-af-feats style), GPU-vectorized.

Key speedup:
  1. Vectorized along the CTC state lattice L (eliminating slow nested Python loops).
  2. Batched across all B substitution candidates in parallel on GPU tensor cores.
  3. Single-pass forward algorithm directly on CUDA.
"""
import torch
import torch.nn.functional as F


def ctc_forward_batch_norm(params, seqmat, blank=0):
    """Vectorized scaled-forward batched across all B sequences.
    params: [P, T] (emission probabilities)
    seqmat: [B, S] (token sequences)
    """
    P, T = params.shape
    B, S = seqmat.shape
    L = 2 * S + 1
    device = params.device

    tok = torch.full((B, L), blank, dtype=torch.long, device=device)
    odd_idx = torch.arange(1, L, 2, device=device)
    tok[:, odd_idx] = seqmat

    can_skip_blank = torch.zeros((B, L), dtype=torch.bool, device=device)
    if S > 1:
        same_tok = (seqmat[:, 1:] == seqmat[:, :-1])  # [B, S-1]
        can_skip_blank[:, 3::2] = ~same_tok

    alphas = torch.zeros((B, L), dtype=torch.float64, device=device)
    alpha_bar = torch.zeros((B, T), dtype=torch.float64, device=device)

    alphas[:, 0] = params[blank, 0]
    alphas[:, 1] = params[seqmat[:, 0], 0]
    alpha_bar[:, 0] = torch.clamp_min(alphas.sum(dim=1), 1e-300)
    alphas = alphas / alpha_bar[:, 0:1]

    emit_all = params[tok, :]  # [B, L, T]

    for t in range(1, T):
        v0 = alphas
        v1 = F.pad(alphas[:, :-1], (1, 0), value=0.0)
        v2_raw = F.pad(alphas[:, :-2], (2, 0), value=0.0)
        v2 = torch.where(can_skip_blank, v2_raw, 0.0)

        curr = (v0 + v1 + v2) * emit_all[:, :, t]

        bar = torch.clamp_min(curr.sum(dim=1), 1e-300)
        alpha_bar[:, t] = bar
        alphas = curr / bar.unsqueeze(1)

    return -torch.log(alpha_bar).sum(dim=1)


def canonical_occupancy(params, labels, blank=0):
    """Vectorized Per-phone occupancy on canonical sequence."""
    P, T = params.shape
    S = labels.shape[0]
    labels = labels.long()
    L = 2 * S + 1
    device = params.device
    logp = torch.log(params.clamp_min(1e-300))  # [P, T]
    NEG = -1e30

    tok = torch.tensor([blank if s % 2 == 0 else int(labels[(s - 1) // 2]) for s in range(L)],
                       dtype=torch.long, device=device)

    can_skip = torch.zeros(L, dtype=torch.bool, device=device)
    if S > 1:
        for s in range(3, L, 2):
            if labels[(s - 1) // 2] != labels[(s - 3) // 2]:
                can_skip[s] = True

    la = torch.full((L, T), NEG, dtype=torch.float64, device=device)
    lb = torch.full((L, T), NEG, dtype=torch.float64, device=device)

    la[0, 0] = logp[blank, 0]
    la[1, 0] = logp[labels[0], 0]
    for t in range(1, T):
        v0 = la[:, t - 1]
        v1 = F.pad(la[:-1, t - 1], (1, 0), value=NEG)
        v2 = F.pad(la[:-2, t - 1], (2, 0), value=NEG)
        v2 = torch.where(can_skip, v2, NEG)
        stacked = torch.stack([v0, v1, v2], dim=0)
        la[:, t] = torch.logsumexp(stacked, dim=0) + logp[tok, t]

    lb[L - 1, T - 1] = 0.0
    lb[L - 2, T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        v0 = lb[:, t + 1] + logp[tok, t + 1]
        v1 = F.pad(lb[1:, t + 1] + logp[tok[1:], t + 1], (0, 1), value=NEG)
        v2 = F.pad(lb[2:, t + 1] + logp[tok[2:], t + 1], (0, 2), value=NEG)
        v2 = torch.where(can_skip, v2, NEG)
        stacked = torch.stack([v0, v1, v2], dim=0)
        lb[:, t] = torch.logsumexp(stacked, dim=0)

    logZ = torch.logsumexp(torch.stack([la[L - 1, T - 1], la[L - 2, T - 1]]), dim=0)
    gamma = torch.exp(la + lb - logZ)
    odd_idx = torch.arange(1, L, 2, device=device)
    occ = gamma[odd_idx].sum(dim=1)
    return occ


def extract_utt_feats_norm_fast(params, labels, blank=0, occ=True, cap_elems=4e7):
    """Vectorized high-throughput GOP feature-vector extraction on GPU/CPU."""
    P, T = params.shape
    S = labels.shape[0]
    device = params.device
    labels = labels.long().to(device)

    nll_canon = ctc_forward_batch_norm(params, labels.view(1, -1), blank=blank)[0]

    feats = torch.zeros((S, 1 + P), dtype=torch.float64, device=device)
    feats[:, 0] = nll_canon

    per_pos = P * (2 * S + 1) * T
    chunk = max(1, int(cap_elems // max(per_pos, 1)))
    for a in range(0, S, chunk):
        b = min(S, a + chunk)
        idx = list(range(a, b))
        seqmat = labels.view(1, -1).repeat(len(idx) * P, 1)
        for k, i in enumerate(idx):
            seqmat[k * P:(k + 1) * P, i] = torch.arange(P, device=device)
        nll = ctc_forward_batch_norm(params, seqmat, blank=blank).view(len(idx), P)
        feats[a:b, 1:] = -nll_canon + nll

    # Deletions (all length S-1)
    if S > 1:
        del_mat = torch.zeros((S, S - 1), dtype=torch.long, device=device)
        for i in range(S):
            del_mat[i] = torch.cat([labels[:i], labels[i + 1:]])
        nll_del = ctc_forward_batch_norm(params, del_mat, blank=blank)
        feats[:, 1 + blank] = -nll_canon + nll_del

    occ_out = None
    if occ:
        occ_out = canonical_occupancy(params, labels, blank=blank)

    return feats.cpu(), (occ_out.cpu() if occ_out is not None else None)
