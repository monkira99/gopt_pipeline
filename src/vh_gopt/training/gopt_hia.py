"""
HIA — Residual Hierarchical Interactive Attention (arXiv:2601.01745, AAAI 2026),
adapted to our CTC-GOP 41-d features. GOP-only, tiny model (embed 48, depth 3),
same output interface as gopt_model.GOPT: forward(x, phn) -> {utt, phone, word}.

Pipeline (paper Fig 2/3, Eq 5-11):
  1. GOP feature + canonical phone one-hot + pos -> Transformer encoder -> X [B,N,D]
  2. Interactive Attention Module: 3 granularity queries -> SelfAttn (bidirectional
     phone<->word<->utt) -> CrossAttn to X -> FFN -> H^phn/H^word/H^utt  [B,D] each
  3. Residual hierarchical:
       phone: S^phn = Conv(X + H^phn);                       phone head
       word : X^word = X + S^phn + H^word; S^word = AspectAttn(X^word); Conv; word heads
       utt  : X^utt = X + S^word + H^utt;  S^utt = TransDecoder(Q^utt, X^utt); utt heads
Heads dropped vs paper (out of scope): word-stress, utt-prosodic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vh_gopt.training.gopt_model import UTT_HEADS, WORD_HEADS


class Conv1dRefine(nn.Module):
    """1-D conv over the phone axis (kernel 5) capturing local context, + residual."""
    def __init__(self, dim, kernel=5):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.act = nn.GELU()

    def forward(self, x):                                    # x [B,N,D]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.act(y)


class HIA(nn.Module):
    def __init__(self, input_dim=41, embed_dim=48, num_heads=4, depth=3,
                 n_phn_cls=40, max_len=50, dropout=0.1,
                 n_utt_head=len(UTT_HEADS), n_word_head=len(WORD_HEADS)):
        super().__init__()
        assert embed_dim % num_heads == 0
        D = embed_dim
        self.max_len = max_len
        self.n_phn_cls = n_phn_cls
        self.n_utt_head = n_utt_head

        # --- 1. encoder ---
        self.in_proj = nn.Linear(input_dim, D)
        self.phn_proj = nn.Linear(n_phn_cls, D)
        self.pos = nn.Parameter(torch.zeros(1, max_len, D)); nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(D, num_heads, dim_feedforward=D * 4, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.enc_norm = nn.LayerNorm(D)

        # --- 2. interactive attention module (3 granularity queries) ---
        self.q_proj = nn.ModuleList([nn.Linear(D, D) for _ in range(3)])   # phn/word/utt queries
        self.self_attn = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
        self.iam_ffn = nn.Sequential(nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D))
        self.iam_norm = nn.LayerNorm(D)

        # --- 3a. phone level ---
        self.phn_conv = Conv1dRefine(D)
        self.phn_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

        # --- 3b. word level ---
        self.word_aspect = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
        self.word_conv = Conv1dRefine(D)
        self.word_head = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
                                        for _ in range(n_word_head)])

        # --- 3c. utterance level ---
        self.utt_query = nn.Parameter(torch.zeros(1, n_utt_head, D)); nn.init.trunc_normal_(self.utt_query, std=0.02)
        dec = nn.TransformerDecoderLayer(D, num_heads, dim_feedforward=D * 4, dropout=dropout,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.utt_dec = nn.TransformerDecoder(dec, 1)
        self.utt_head = nn.ModuleList([nn.Linear(D, 1) for _ in range(n_utt_head)])

    def forward(self, x, phn):
        B = x.size(0)
        pad = (phn < 0)                                           # [B,N]
        phn_idx = (phn + 1).clamp(min=0).long()
        oh = F.one_hot(phn_idx, num_classes=self.n_phn_cls).float()
        h = self.in_proj(x) + self.phn_proj(oh) + self.pos       # [B,N,D]
        X = self.enc_norm(self.enc(h, src_key_padding_mask=pad)) # phone embeddings

        # masked mean pool for query init
        keep = (~pad).unsqueeze(-1).float()
        pooled = (X * keep).sum(1) / keep.sum(1).clamp_min(1)    # [B,D]

        # IAM: 3 granularity queries -> self-attn (cross-granularity) -> cross-attn to X -> FFN
        Q = torch.stack([proj(pooled) for proj in self.q_proj], dim=1)   # [B,3,D]
        Qs, _ = self.self_attn(Q, Q, Q)
        Qc, _ = self.cross_attn(Qs, X, X, key_padding_mask=pad)
        Hgr = self.iam_norm(Qc + self.iam_ffn(Qc))               # [B,3,D]
        H_phn, H_word, H_utt = Hgr[:, 0:1], Hgr[:, 1:2], Hgr[:, 2:3]     # [B,1,D] broadcast over N

        # phone level
        S_phn = self.phn_conv(X + H_phn)                         # [B,N,D]
        phone = self.phn_head(S_phn).squeeze(-1)                 # [B,N]

        # word level
        X_word = X + S_phn + H_word
        S_word, _ = self.word_aspect(X_word, X_word, X_word, key_padding_mask=pad)
        S_word = self.word_conv(X_word + S_word)
        word = torch.cat([head(S_word) for head in self.word_head], dim=-1)   # [B,N,2]

        # utterance level
        X_utt = X + S_word + H_utt
        Qu = self.utt_query.expand(B, -1, -1)                    # [B,U,D]
        S_utt = self.utt_dec(Qu, X_utt, memory_key_padding_mask=pad)          # [B,U,D]
        utt = torch.cat([head(S_utt[:, i]) for i, head in enumerate(self.utt_head)], dim=1)  # [B,U]

        return {"utt": utt, "phone": phone, "word": word}


if __name__ == "__main__":
    m = HIA(embed_dim=48, num_heads=4, depth=3)
    x = torch.randn(2, 50, 41); phn = torch.randint(0, 39, (2, 50)); phn[:, 30:] = -1
    o = m(x, phn)
    print({k: tuple(v.shape) for k, v in o.items()}, "params:", sum(p.numel() for p in m.parameters()))
