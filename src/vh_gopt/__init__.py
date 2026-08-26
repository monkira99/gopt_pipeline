"""VH-GOPT: VuiHoc Goodness-of-Pronunciation pipeline.

Subpackages:
  core      - thu vien doc lap: G2P, parse/align vendor, GOP feats (KoelLabs CTC)
  dataset   - corpus snapshot/fetch -> pack npz -> verify -> push HuggingFace (datasets)
  training  - GOPT/HIA scoring model + trainer
"""

__version__ = "0.1.0"
