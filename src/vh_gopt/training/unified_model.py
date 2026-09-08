"""Mô hình HỢP NHẤT: GOPT scorer + MSDD detector trong MỘT trunk.

Động cơ (câu hỏi thực nghiệm đã xác nhận trên nhãn v2, fit-val/test-test):
  - Gộp 2 nhiệm vụ KHÔNG làm tụt phone PCC (0.831 vs 0.827 = positive transfer).
  - MSDD ngang/hơn model tách (F1 0.782 vs 0.768; del_recall 0.674 vs 0.618).
  - "Sai âm nhưng điểm cao" tự rớt 77%->18.7% (đúng sàn nhiễu nhãn 18.3%) CHỈ nhờ trunk chung.
  - Consistency loss là THỪA+HẠI (ép dưới sàn nhãn -> giết sub_recall) => mặc định w_cons=0.

Kiến trúc = GOPT nguyên vẹn (giữ mọi lever: arch/wavlm_fuse/attn_pool/utt_prosody/no_word_head)
+ 2 head token-level đọc CHÍNH `tok` (embedding token phone) mà phn_head đang đọc:
  - det_head : 3 lớp (0=OK, 1=Sub, 2=Del)
  - diag_head: 39 lớp ARPA39 (âm thay thế)
Không thêm tham số acoustic; WavLM đã chạy sẵn cho scorer -> serve MSDD ~0 chi phí thêm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vh_gopt.training.gopt_model import GOPT, WORD_HEADS

PHONE_NUM = 39


class GOPTUnified(GOPT):
    """GOPT + head det(3) + diag(39) trên token phone. Trả thêm 'det','diag'; bỏ 'tok'."""
    def __init__(self, *a, det_hidden=None, **k):
        super().__init__(*a, **k)
        ed = self.norm.normalized_shape[0]        # embed_dim
        h = det_hidden or ed
        self.det_head = nn.Sequential(nn.Linear(ed, h), nn.GELU(),
                                      nn.Dropout(0.1), nn.Linear(h, 3))
        self.diag_head = nn.Sequential(nn.Linear(ed, h), nn.GELU(),
                                       nn.Dropout(0.1), nn.Linear(h, PHONE_NUM))

    def forward(self, x, phn):
        out = super().forward(x, phn)
        tok = out.pop("tok")                       # [B,L,D] — không để lọt vào predictions
        out["det"] = self.det_head(tok)            # [B,L,3]
        out["diag"] = self.diag_head(tok)          # [B,L,39]
        return out


class GOPTUnifiedForScoring(GOPTUnified):
    """GOPTUnified + loss hợp nhất (HF-Trainer compatible).

    loss = w_phn*MSE_phone + w_word*MSE_word + w_utt*MSE_utt        (scorer, GOPT-faithful)
         + w_det*CE_det(class-weighted) + w_diag*CE_diag             (MSDD)
         + w_cons*consistency                                        (mặc định 0)

    consistency (mềm, một phía): phạt khi điểm cao ĐỒNG THỜI P(lỗi) cao, tức
      relu(phone/100 - P(correct))  với P(correct)=softmax(det)[OK].
      Đã chứng minh THỪA (trunk chung tự đạt sàn nhãn) -> để w_cons=0; giữ tùy chọn để tái kiểm.
    """
    def __init__(self, *a, w_phn=1.0, w_word=1.0, w_utt=1.0, noise=0.0,
                 w_det=1.0, w_diag=1.0, w_cons=0.0, det_class_w=None,
                 score_scale=100.0, score_norm=None, **k):
        super().__init__(*a, **k)
        self.w_phn, self.w_word, self.w_utt, self.noise = w_phn, w_word, w_utt, noise
        self.w_det, self.w_diag, self.w_cons = w_det, w_diag, w_cons
        self.score_scale = score_scale
        # Cân bằng nhiệm vụ: MSE điểm (thang 0-100, ~O(1e3)) áp đảo CE (~O(1)) trong trunk chung
        # -> chia MSE cho score_norm để hai họ loss cùng ~O(1) (khớp balance đã validate ở probe: /300 @ scale100).
        self.score_norm = score_norm if score_norm is not None else 300.0 * (score_scale / 100.0) ** 2
        # trọng số lớp detection: persistent=False -> KHÔNG lưu vào safetensors (tránh key lạ khi reload).
        if det_class_w is not None:
            self.register_buffer("det_class_w", torch.tensor(det_class_w, dtype=torch.float32),
                                 persistent=False)
        else:
            self.det_class_w = None

    def forward(self, feat, phn, phone_label=None, word_label=None, utt_label=None,
                msdd_type=None, msdd_sub=None):
        if self.training and self.noise > 0:
            feat = feat + (torch.rand_like(feat) - 1) * self.noise
        out = super().forward(feat, phn)
        res = {"phone": out["phone"], "word": out["word"], "utt": out["utt"],
               "det": out["det"], "diag": out["diag"]}
        if phone_label is not None:
            # ---- scorer (GOPT-faithful masked MSE) ----
            m = (phone_label >= 0).float()
            wl = word_label[..., :len(WORD_HEADS)]
            mw = (wl >= 0).float()
            lp = (((out["phone"] - phone_label.clamp(min=0)) ** 2) * m).sum() / m.sum().clamp_min(1)
            lw = (((out["word"] - wl.clamp(min=0)) ** 2) * mw).sum() / mw.sum().clamp_min(1)
            mu = (utt_label >= 0).float()
            lu = (((out["utt"] - utt_label) ** 2) * mu).sum() / mu.sum().clamp_min(1)
            loss = (self.w_phn * lp + self.w_word * lw + self.w_utt * lu) / self.score_norm
            # ---- MSDD ----
            if msdd_type is not None:
                l_det = F.cross_entropy(out["det"].reshape(-1, 3), msdd_type.reshape(-1),
                                        weight=self.det_class_w, ignore_index=-1)
                l_diag = F.cross_entropy(out["diag"].reshape(-1, PHONE_NUM), msdd_sub.reshape(-1),
                                         ignore_index=-1)
                loss = loss + self.w_det * l_det + self.w_diag * l_diag
                # ---- consistency (tùy chọn, mặc định tắt) ----
                if self.w_cons > 0:
                    pok = torch.softmax(out["det"], dim=-1)[..., 0]              # P(OK)
                    s01 = (out["phone"] / self.score_scale).clamp(0, 1)
                    viol = torch.relu(s01 - pok) * m
                    loss = loss + self.w_cons * (viol.sum() / m.sum().clamp_min(1))
            res["loss"] = loss
        return res
