"""
GOPT adapted for CTC-GOP 41-d features, with optional JCAPT (arXiv:2506.19315)
components that apply to a GOP-only Transformer (no SSL fusion / no Mamba):
  - use_phono : concat articulatory attribute vector to the phone one-hot (§2.2)
  - n_think   : learnable "think tokens" appended to the sequence (§2.3)
  - attn_pool : per-aspect attention pooling for utterance heads instead of CLS (§2.4)
  - arch      : base | mlp | concat | film  (phone-conditioning / input projection)

Faithful to YuanGongND/gopt otherwise: fixed max_len (50) transformer over phones
with learnable CLS tokens for utterance heads (when attn_pool is off).

Input:
  x   : [B, max_len, input_dim]  GOP feature per phone (0-padded)
  phn : [B, max_len]             phone class id in [0, n_phn_cls-2], pad = -1
Output (dict): utt [B,U], phone [B,L], word [B,L,2]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

UTT_HEADS = ("accuracy", "completeness", "fluency", "total")
# completeness is near-constant on SO762 (4975/5000=10) so its PCC is junk, but keeping the
# head as an auxiliary task regularizes the shared encoder and lifts phone/word — so we KEEP
# the head and merely exclude completeness from the reported `mean` (see MEAN_HEADS).
MEAN_HEADS = ("accuracy", "fluency", "total")
WORD_HEADS = ("accuracy",)


class GOPT(nn.Module):
    def __init__(self, input_dim=41, embed_dim=48, num_heads=4, depth=3,
                 n_phn_cls=40, max_len=150, dropout=0.1, arch="base",
                 use_phono=False, phono_matrix=None, n_think=0, attn_pool=False,
                 utt_prosody=False, prosody_dim=0,
                 wavlm_dim=0, wavlm_fuse="stack",
                 n_utt_head=len(UTT_HEADS), n_word_head=len(WORD_HEADS)):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert arch in ("base", "mlp", "concat", "film"), arch
        assert wavlm_fuse in ("stack", "phone", "utt"), wavlm_fuse
        self.max_len = max_len
        self.n_phn_cls = n_phn_cls
        self.n_utt_head = n_utt_head
        self.arch = arch
        self.use_phono = use_phono
        self.n_think = n_think
        self.attn_pool = attn_pool
        # prosody fed ONLY to the utterance branch (phone/word heads stay untouched).
        # x is [feat_dim(=input_dim) | prosody_dim]; encoder consumes only the first feat_dim.
        self.utt_prosody = utt_prosody
        self.prosody_dim = prosody_dim
        self.feat_dim = input_dim                          # prosody bắt đầu ở offset này (data: [core|wavlm|prosody])
        # WavLM fusion: 'stack' = nhồi chung core (như cũ); 'phone'/'utt' = tách slice + projection riêng.
        self.wavlm_dim = wavlm_dim
        self.wavlm_fuse = wavlm_fuse
        sep_wavlm = wavlm_dim > 0 and wavlm_fuse != "stack"
        core_dim = (input_dim - wavlm_dim) if sep_wavlm else input_dim   # phần GOP+occ vào nhánh phone
        self.core_dim = core_dim

        # phone symbolic vector = one-hot [+ articulatory attributes]
        if use_phono:
            assert phono_matrix is not None, "use_phono needs phono_matrix"
            self.register_buffer("phono", torch.tensor(phono_matrix, dtype=torch.float32))
            sym_dim = n_phn_cls + self.phono.size(1)
        else:
            sym_dim = n_phn_cls
        self.sym_dim = sym_dim

        # GOP feature projection + phone conditioning (chỉ nhận core_dim = GOP+occ khi tách WavLM)
        if arch == "concat":
            self.in_proj = nn.Linear(core_dim + sym_dim, embed_dim)
        elif arch == "mlp":
            self.in_proj = nn.Sequential(nn.Linear(core_dim, embed_dim), nn.GELU(),
                                         nn.Linear(embed_dim, embed_dim))
            self.phn_proj = nn.Linear(sym_dim, embed_dim)
        elif arch == "film":
            self.in_proj = nn.Linear(core_dim, embed_dim)
            self.film = nn.Linear(sym_dim, 2 * embed_dim)
        else:
            self.in_proj = nn.Linear(core_dim, embed_dim)
            self.phn_proj = nn.Linear(sym_dim, embed_dim)

        # WavLM projection riêng: 'phone' -> cộng vào token phone; 'utt' -> pool(mean+std) chỉ vào nhánh utt
        if sep_wavlm and wavlm_fuse == "phone":
            self.wavlm_proj = nn.Linear(wavlm_dim, embed_dim)
        elif sep_wavlm and wavlm_fuse == "utt":
            self.wavlm_utt = nn.Sequential(nn.Linear(2 * wavlm_dim, embed_dim), nn.GELU(),
                                           nn.Linear(embed_dim, embed_dim))

        n_cls = 0 if attn_pool else n_utt_head
        self.n_cls = n_cls
        if n_cls:
            self.cls = nn.Parameter(torch.zeros(1, n_cls, embed_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)
        if n_think:
            self.think = nn.Parameter(torch.zeros(1, n_think, embed_dim))
            nn.init.trunc_normal_(self.think, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, n_cls + max_len + n_think, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)

        # per-aspect attention pooling (JCAPT §2.4) for utterance heads
        if attn_pool:
            self.pool_W = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_utt_head)])
            self.pool_w = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(n_utt_head)])

        # utterance-level prosody summary (mean+std pooled over valid phones -> embed_dim)
        if utt_prosody:
            assert prosody_dim > 0, "utt_prosody needs prosody_dim>0"
            self.pros_proj = nn.Sequential(nn.Linear(2 * prosody_dim, embed_dim), nn.GELU(),
                                           nn.Linear(embed_dim, embed_dim))

        self.utt_head = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(n_utt_head)])
        self.phn_head = nn.Linear(embed_dim, 1)
        self.word_head = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(n_word_head)])

    def _sym(self, phn_idx):
        oh = F.one_hot(phn_idx, num_classes=self.n_phn_cls).float()
        if self.use_phono:
            return torch.cat([oh, self.phono[phn_idx]], dim=-1)
        return oh

    def forward(self, x, phn):
        B = x.size(0)
        pad = (phn < 0)                                   # [B, L] True where padded
        phn_idx = (phn + 1).clamp(min=0).long()           # pad -> 0, phones -> 1..n
        sym = self._sym(phn_idx)                          # [B, L, sym_dim]

        # peel prosody off the tail; encoder never sees it (phone/word branch unchanged)
        e_pros = None
        if self.utt_prosody:
            pros = x[..., self.feat_dim:self.feat_dim + self.prosody_dim]   # [B, L, P]
            x = x[..., :self.feat_dim]
            keep = (~pad).unsqueeze(-1).float()                            # [B, L, 1]
            n = keep.sum(1).clamp_min(1)                                   # [B, 1]
            mean = (pros * keep).sum(1) / n                                # [B, P]
            var = ((pros - mean.unsqueeze(1)) ** 2 * keep).sum(1) / n
            stats = torch.cat([mean, var.clamp_min(0).sqrt()], dim=-1)     # [B, 2P]
            e_pros = self.pros_proj(stats)                                 # [B, D]

        # tách WavLM khỏi core (chỉ khi fuse != stack). x còn lại = GOP+occ cho nhánh phone.
        e_wavlm, wl = None, None
        if self.wavlm_dim > 0 and self.wavlm_fuse != "stack":
            wl = x[..., self.core_dim:self.core_dim + self.wavlm_dim]      # [B, L, W]
            x = x[..., :self.core_dim]
            if self.wavlm_fuse == "utt":                                   # pool mean+std -> chỉ nhánh utt
                keep = (~pad).unsqueeze(-1).float()
                n = keep.sum(1).clamp_min(1)
                mean = (wl * keep).sum(1) / n
                var = ((wl - mean.unsqueeze(1)) ** 2 * keep).sum(1) / n
                e_wavlm = self.wavlm_utt(torch.cat([mean, var.clamp_min(0).sqrt()], dim=-1))  # [B, D]

        if self.arch == "concat":
            h = self.in_proj(torch.cat([x, sym], dim=-1))
        elif self.arch == "film":
            gamma, beta = self.film(sym).chunk(2, dim=-1)
            h = self.in_proj(x) * (1 + gamma) + beta
        else:                                             # base, mlp
            h = self.in_proj(x) + self.phn_proj(sym)      # [B, L, D]

        if self.wavlm_fuse == "phone" and wl is not None:  # WavLM projection riêng -> cộng vào token phone
            h = h + self.wavlm_proj(wl)

        parts, mask_parts = [], []
        if self.n_cls:
            parts.append(self.cls.expand(B, -1, -1))
            mask_parts.append(torch.zeros(B, self.n_cls, dtype=torch.bool, device=x.device))
        parts.append(h)
        mask_parts.append(pad)
        if self.n_think:
            parts.append(self.think.expand(B, -1, -1))
            mask_parts.append(torch.zeros(B, self.n_think, dtype=torch.bool, device=x.device))
        h = torch.cat(parts, dim=1) + self.pos
        key_pad = torch.cat(mask_parts, dim=1)

        h = self.norm(self.enc(h, src_key_padding_mask=key_pad))

        tok_out = h[:, self.n_cls:self.n_cls + self.max_len]      # [B, L, D] phone tokens

        if self.attn_pool:                                        # per-aspect attention pool
            neg = torch.zeros(B, self.max_len, device=x.device).masked_fill(pad, float("-inf"))
            utt_cols = []
            for i, head in enumerate(self.utt_head):
                score = self.pool_w[i](torch.tanh(self.pool_W[i](tok_out))).squeeze(-1)  # [B,L]
                alpha = torch.softmax(score + neg, dim=1).unsqueeze(-1)                  # [B,L,1]
                pooled = (alpha * tok_out).sum(1)                                        # [B,D]
                if e_pros is not None:
                    pooled = pooled + e_pros
                if e_wavlm is not None:
                    pooled = pooled + e_wavlm
                utt_cols.append(head(pooled))
            utt = torch.cat(utt_cols, dim=1)
        else:
            cls_out = h[:, :self.n_cls]                            # [B, U, D]
            if e_pros is not None:
                cls_out = cls_out + e_pros.unsqueeze(1)            # broadcast prosody to every utt head
            if e_wavlm is not None:
                cls_out = cls_out + e_wavlm.unsqueeze(1)
            utt = torch.cat([head(cls_out[:, i]) for i, head in enumerate(self.utt_head)], dim=1)

        phone = self.phn_head(tok_out).squeeze(-1)                 # [B, L]
        word = torch.cat([head(tok_out) for head in self.word_head], dim=-1)  # [B, L, n_word_head]
        return {"utt": utt, "phone": phone, "word": word}


if __name__ == "__main__":
    from phono import phono_buffer
    x = torch.randn(2, 50, 41)
    phn = torch.randint(0, 39, (2, 50)); phn[:, 30:] = -1
    pm = phono_buffer(["AA"] * 39, 40)
    for kw in [{}, {"arch": "film"}, {"use_phono": True, "phono_matrix": pm},
               {"n_think": 4}, {"attn_pool": True}]:
        m = GOPT(embed_dim=24, num_heads=1, depth=3, **kw)
        o = m(x, phn)
        print(kw, {k: tuple(v.shape) for k, v in o.items()},
              "params:", sum(p.numel() for p in m.parameters()))
