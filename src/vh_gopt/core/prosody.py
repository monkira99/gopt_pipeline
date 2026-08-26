"""Prosodic features (3M, arXiv:2208.09110 §II.B) computed from the SAME CTC emissions:
  - duration : per-phone segment length (#frames), from CTC Viterbi forced alignment -> 1-d
  - energy   : per-phone RMSE statistics {mean,std,median,mad,sum,max,min}           -> 7-d
"""
import numpy as np
import torch

ENERGY_STATS = ["mean", "std", "median", "mad", "sum", "max", "min"]
ENERGY_DIM = len(ENERGY_STATS)
PROSODY_DIM = 1 + ENERGY_DIM  # 8


def ctc_align(logp, labels, blank=0):
    """logp [T,C] log-probs, labels [S] token ids -> path[T] (CTC state per frame).
    State s: even = blank, odd = label (s-1)//2. Every label state is visited >=1 frame."""
    T, C = logp.shape
    S = len(labels)
    L = 2 * S + 1
    NEG = -1e30
    tok = [blank if s % 2 == 0 else int(labels[(s - 1) // 2]) for s in range(L)]

    delta = np.full((T, L), NEG)
    back = np.zeros((T, L), np.int32)
    lp = logp.numpy() if torch.is_tensor(logp) else np.asarray(logp)
    delta[0, 0] = lp[0, blank]
    delta[0, 1] = lp[0, tok[1]]
    for t in range(1, T):
        for s in range(L):
            best_v = delta[t - 1, s]
            best_s = s
            if s > 0 and delta[t - 1, s - 1] > best_v:
                best_v = delta[t - 1, s - 1]
                best_s = s - 1
            if s > 1 and s % 2 == 1 and tok[s] != tok[s - 2] and delta[t - 1, s - 2] > best_v:
                best_v = delta[t - 1, s - 2]
                best_s = s - 2
            delta[t, s] = best_v + lp[t, tok[s]]
            back[t, s] = best_s
    last = L - 1 if delta[T - 1, L - 1] >= delta[T - 1, L - 2] else L - 2
    path = np.zeros(T, np.int32)
    path[T - 1] = last
    for t in range(T - 1, 0, -1):
        last = back[t, last]
        path[t - 1] = last
    return path


def frame_rmse(wav, T):
    """RMSE energy per emission frame: split waveform into T uniform segments."""
    N = len(wav)
    if N == 0 or T == 0:
        return np.zeros(T, np.float32)
    edges = np.linspace(0, N, T + 1, dtype=int)
    e = np.zeros(T, np.float32)
    for t in range(T):
        seg = wav[edges[t]:edges[t + 1]]
        if len(seg):
            e[t] = float(np.sqrt(np.mean(seg * seg)))
    return e


def _stats(v):
    if len(v) == 0:
        return np.zeros(ENERGY_DIM, np.float32)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return np.array([v.mean(), v.std(), med, mad,
                     v.sum(), v.max(), v.min()], dtype=np.float32)


def phone_segments(logp, labels, blank=0):
    """Tra ve list[(start, end)] - dai frame CTC gan cho tung phone chuan."""
    path = ctc_align(logp, labels, blank=blank)
    T = len(path)
    S = len(labels)
    label_states = [2 * s + 1 for s in range(S)]

    first = []
    for s_target in label_states:
        idx = np.where(path == s_target)[0]
        first.append(int(idx[0]) if len(idx) else 0)

    segs = []
    for s in range(S):
        a = first[s]
        b = first[s + 1] if (s + 1 < S) else T
        if b <= a:
            b = a + 1
        segs.append((a, b))
    return segs, T


def phone_prosody(logp, labels, wav, blank=0):
    """Returns duration [S] (frames) and energy [S, 7] for the canonical phones."""
    segs, T = phone_segments(logp, labels, blank=blank)
    S = len(labels)
    dur = np.zeros(S, np.float32)
    eng = np.zeros((S, ENERGY_DIM), np.float32)

    e_frame = frame_rmse(wav, T)
    for s, (a, b) in enumerate(segs):
        dur[s] = float(b - a)
        eng[s] = _stats(e_frame[a:b])
    return dur, eng
