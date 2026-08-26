# vh-gopt — Pipeline dataset + training GOPT/MSDD (VuiHoc)

Package tu chua (submodule), trien khai duoc tren server khac: build dataset
tu corpus snapshot → kiem chat luong → push HuggingFace o dang
`datasets.DatasetDict` (Arrow, `load_dataset` duoc) → training GOPT/HIA.

## cau truc

```
gopt_pipeline/
├── configs/vh_gold.json        # cau hinh mac dinh (path, model, repo HF)
├── run.sh                      # dispatcher: install|snapshot|fetch|pack|verify|push|train
└── src/vh_gopt/
    ├── core/                   # thu vien doc lap: G2P, parse/align 3 vendor,
    │                           # CTC-GOP feats 80-d (KoelLabs), ARPA39
    ├── dataset/
    │   ├── snapshot_corpus.py  # (may co du lieu thoi) gom raw -> snapshot + push HF private
    │   ├── fetch_corpus.py     # (server build) tai snapshot ve data/corpus
    │   ├── pack_stage2.py      # corpus -> train/val/test_*.npz (labels + GOP feats)
    │   ├── verify_dataset.py   # cong chat luong: coverage, mask-weight, leakage, feat that
    │   └── push_dataset.py     # npz -> DatasetDict -> push_to_hub (+ card, metadata.json)
    └── training/
        ├── gopt_model.py       # GOPT (Transformer nho, multi-head scoring)
        ├── gopt_hia.py         # bien the HIA
        └── gopt_train.py       # trainer (doc npz trung gian)
```

## Luong chuan

```
(may du lieu)  ./run.sh snapshot --push-repo <org>/gopt-vh-corpus
                     │  (raw manifest + vendor json + audio ~4.5 GB)
(server build) ./run.sh fetch --repo <org>/gopt-vh-corpus
               ./run.sh pack                       # trich GOP 80-d KoelLabs (~1-2s/audio CPU)
               ./run.sh verify                     # PHAI PASS truoc khi push
               ./run.sh push --repo <org>/gopt-vh-scripted-gold
               ./run.sh build --repo <org>/gopt-vh-gold   # GOLD Arrow (audio+labels) -> HF
               ./run.sh train --train data/gopt_vh_scripted_gold/train.npz \
                              --test  data/gopt_vh_scripted_gold/test_unseen_speakers.npz ...
```

## Setup server moi

```bash
git clone <url>/gopt_pipeline.git          # hoac: git submodule add <url> gopt_pipeline
cd gopt_pipeline
python3 -m venv .venv && source .venv/bin/activate
./run.sh install                            # pip install -e .
export HF_TOKEN=hf_xxx                      # can de doc snapshot private + push
# lan dau tien g2p_en se tai NLTK data (can internet)
./run.sh pack --limit 5                     # smoke test nhanh truoc khi chay full
```

Cau hinh: sua `configs/vh_gold.json` (corpus_dir, out_dir, model_id,
hf_corpus_repo, hf_dataset_repo) hoac truyen flag CLI override.

## Dinh dang artifact tren HF

- `push` tao `datasets.DatasetDict{train, val, test_unseen_speakers, test_unseen_prompts}`,
  moi row = 1 audio voi day du tensor phone/word/utt/MSDD + GOP feature.
- `metadata.json` (phone_list ARPA39, utt_heads, max_len, quy uoc mask).
- Mask = `-1` / `-1.0`; trong so tin cay `*_weight` ∈ [0,1].
- Tai lai:

```python
from datasets import load_dataset
ds = load_dataset("<org>/gopt-vh-scripted-gold")
```

`data/*.npz` chi la trung gian noi bo (gitignored). Trainer hien tai doc npz;
wrapper `load_dataset` cho training la viec ke tiep (xem Roadmap).

## Known gaps (ke thua tu audit 2026-08-26)

1. `gopt_train.compute_metrics` chua mask nhan -1 cua utt-fluency va word masked
   → PCC metric bi nhieu nhe; fix truoc khi so sanh checkpoint giua cac run.
2. Trainer chua tieu thu `phone_weight/word_weight/utt_weight` (loss = masked MSE
   thuan); trong so consensus van nam trong npz/dataset de dung sau.
3. MSDD detection: vi tri 0-vendor bi gan mac dinh Sub; Del chi can 1 phieu —
   chap nhan duoc cho Stage 2, can sua neu train detection nghiem tuc.

## Roadmap

- [ ] `vh_gopt.training.data`: loader doc truc tiep `load_dataset(...)` thay npz
- [ ] Wire consensus weights vao loss (homoscedastic 1/sigma^2 theo bao cao)
- [ ] CI: verify_dataset chay trong CI sau moi commit dataset
