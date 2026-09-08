"""Gom metrics.json của grid unified -> mean±std theo (nhóm, split). So base vs +occ."""
import json, glob, os, numpy as np
from collections import defaultdict

KEYS = ["phone", "mean", "det_f1", "del_recall", "sub_recall", "diag_acc", "pct_flag_hi"]
SPLITS = ["test_unseen_speakers", "test_unseen_prompts"]
acc = defaultdict(lambda: defaultdict(list))   # (group, split) -> key -> [vals]

for mp in sorted(glob.glob("ckpt/*/metrics.json")):
    tag = os.path.basename(os.path.dirname(mp))
    if not (tag.startswith("base_s") or tag.startswith("occ_s")):
        continue
    grp = "base" if tag.startswith("base_s") else "+occ"
    m = json.load(open(mp))
    for sp in SPLITS:
        if sp in m:
            for k in KEYS:
                if k in m[sp]:
                    acc[(grp, sp)][k].append(m[sp][k])

def fmt(vals):
    a = np.array(vals, float)
    return f"{a.mean():.3f}±{a.std():.3f}" if len(a) > 1 else (f"{a[0]:.3f}" if len(a) else "-")

for sp in SPLITS:
    print(f"\n================ {sp} ================")
    print(f"{'group':6s} " + " ".join(f"{k:>14s}" for k in KEYS))
    for grp in ["base", "+occ"]:
        d = acc[(grp, sp)]
        n = len(next(iter(d.values()))) if d else 0
        print(f"{grp:6s} " + " ".join(f"{fmt(d.get(k, [])):>14s}" for k in KEYS) + f"   (n={n})")
