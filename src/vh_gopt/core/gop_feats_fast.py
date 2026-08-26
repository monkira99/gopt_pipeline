"""Fast GOP feature-vector extraction (is24 gop-af-feats style), GPU FP32 Tensor Core accelerated.

Key speedup:
  1. TorchScript JIT compiled CTC Forward algorithm (eliminating Python kernel launch overhead).
  2. Batched across all B substitution candidates in a single high-throughput pass on GPU.
  3. Single-pass forward algorithm directly on CUDA.
"""
import torch
import torch.nn.functional as F


@torch.jit.script
def _ctc_forward_batch_norm_jit(params: torch.Tensor, seqmat: torch.Tensor, blank: int = 0) -> torch.Tensor:
    """Vectorized scaled-forward batched across all B sequences on CUDA FP32."""
    P = params.size(0)
    T = params.size(1)
    B = seqmat.size(0)
    S = seqmat.size(1)
    L = 2 * S + 1
    device = params.device
    dtype = params.dtype

    tok = torch.full((B, L), blank, dtype=torch.long, device=device)
    for s in range(S):
        tok[:, 2 * s + 1] = seqmat[:, s]

    can_skip_blank = torch.zeros((B, L), dtype=torch.bool, device=device)
    for s in range(1, S):
        same = (seqmat[:, s] == seqmat[:, s - 1])
        can_skip_blank[:, 2 * s + 1] = ~same

    alphas = torch.zeros((B, L), dtype=dtype, device=device)
    alpha_bar = torch.zeros((B, T), dtype=dtype, device=device)

    alphas[:, 0] = params[blank, 0]
    alphas[:, 1] = params[seqmat[:, 0], 0]
    bar0 = torch.clamp_min(alphas.sum(dim=1), 1e-20)
    alpha_bar[:, 0] = bar0
    alphas = alphas / bar0.unsqueeze(1)

    emit_all = params[tok, :]

    for t in range(1, T):
        v0 = alphas
        v1 = F.pad(alphas[:, :-1], (1, 0), value=0.0)
        v2_raw = F.pad(alphas[:, :-2], (2, 0), value=0.0)
        v2 = torch.where(can_skip_blank, v2_raw, torch.zeros_like(v2_raw))

        curr = (v0 + v1 + v2) * emit_all[:, :, t]

        bar = torch.clamp_min(curr.sum(dim=1), 1e-20)
        alpha_bar[:, t] = bar
        alphas = curr / bar.unsqueeze(1)

    return -torch.log(alpha_bar).sum(dim=1)


def ctc_forward_batch_norm(params: torch.Tensor, seqmat: torch.Tensor, blank: int = 0) -> torch.Tensor:
    return _ctc_forward_batch_norm_jit(params, seqmat, blank)


def extract_utt_feats_norm_fast(params: torch.Tensor, labels: torch.Tensor, blank: int = 0, occ: bool = False, cap_elems: float = 5e8):
    """Vectorized high-throughput GOP feature-vector extraction on GPU/CPU (FP32).
    cap_elems=5e8 cho phép gom toàn bộ candidate của câu vào 1 pass duy nhất."""
    P, T = params.shape
    S = labels.shape[0]
    device = params.device
    dtype = params.dtype
    labels = labels.long().to(device)

    nll_canon = ctc_forward_batch_norm(params, labels.view(1, -1), blank=blank)[0]

    feats = torch.zeros((S, 1 + P), dtype=dtype, device=device)
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

    return feats.cpu(), None
