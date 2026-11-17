"""Consistency checks on an output folder (local or a snapshot_download of the HF repo),
plus a smoke-test probe: leave-one-session-out logistic regression on PHQ8_Binary per
layer at segment level. With a handful of sessions the AUCs only show that the pipeline
works end to end, not that the results mean anything.

    python scripts/validate.py outputs/sample/hf
    python scripts/validate.py --hf GenerationGenerative/daic-woz-embeddings-sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from safetensors.numpy import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab.store import read_layer  # noqa: E402

LAYERED = ["whisper_small/pooled.parquet", "hubert_base/pooled.parquet", "text/roberta_base.parquet"]


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        check.failed = True


check.failed = False


def loso_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    scores = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
        clf.fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--hf")
    args = ap.parse_args()
    if args.hf:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(args.hf, repo_type="dataset"))
    else:
        root = Path(args.root)
    print(f"validating {root}")

    seg = pd.read_parquet(root / "segments.parquet")
    ids = seg.segment_id.tolist()
    check(seg.segment_id.is_unique, f"segments.parquet: {len(seg)} unique segment_ids")
    check(bool((seg.dur_s >= 1.0 - 1e-6).all() and (seg.dur_s <= 30.0 + 1e-6).all()), "segment durations in [1, 30] s")

    for rel in LAYERED:
        t = pq.read_table(root / rel)
        check(t.column("segment_id").to_pylist() == ids, f"{rel}: rows aligned with segments ({t.num_rows})")
        layer_cols = [c for c in t.column_names if c != "segment_id"]
        bad = 0
        for c in layer_cols:
            a = t.column(c).combine_chunks().flatten().to_numpy()
            bad += int((~np.isfinite(a)).any())
        check(bad == 0, f"{rel}: {len(layer_cols)} columns, all finite in float16")
    for rel in ["egemaps/functionals.parquet", "text/mpnet.parquet"]:
        t = pq.read_table(root / rel)
        check(t.column("segment_id").to_pylist() == ids, f"{rel}: rows aligned with segments")
    eg = pd.read_parquet(root / "egemaps/functionals.parquet")
    check(eg.shape[1] - 1 == 88 and bool(np.isfinite(eg.drop(columns="segment_id").values).all()),
          f"egemaps: {eg.shape[1] - 1} functionals, all finite")

    # frames vs pooled mean for one segment per session
    man = json.loads((root / "manifest.json").read_text())
    layers = man["config"]["models"]["frame_layers"]
    for model in ["whisper_small", "hubert_base"]:
        worst = 0.0
        for sess, sid in seg.groupby("session_id").segment_id.first().items():
            fr = load_file(root / model / "frames" / f"{sess}.safetensors")
            n_expected = int((seg.session_id == sess).sum()) * len(layers)
            if len(fr) != n_expected:
                check(False, f"{model}/frames/{sess}: {len(fr)} arrays, expected {n_expected}")
            for k in layers:
                _, m = read_layer(root / model / "pooled.parquet", "mean", k)
                pooled = m[ids.index(sid)]
                frames = fr[f"{sid}/layer_{k:02d}"].astype(np.float32)
                worst = max(worst, float(np.abs(frames.mean(0) - pooled).max() / (np.abs(pooled).max() + 1e-6)))
        check(worst < 0.02, f"{model}: frame mean matches pooled mean (max rel err {worst:.4f})")

    # frame counts: whisper trims to ceil(dur*50); hubert ~ dur*50 (conv receptive field)
    sid = ids[0]
    d = seg.dur_s.iloc[0]
    wf = load_file(root / "whisper_small/frames" / f"{seg.session_id.iloc[0]}.safetensors")[f"{sid}/layer_00"]
    hf = load_file(root / "hubert_base/frames" / f"{seg.session_id.iloc[0]}.safetensors")[f"{sid}/layer_00"]
    check(abs(wf.shape[0] - d * 50) <= 1 and abs(hf.shape[0] - d * 50) <= 1,
          f"frame counts for {d:.2f}s: whisper {wf.shape[0]}, hubert {hf.shape[0]}")

    print("\nsmoke-test probe: LOSO segment-level AUC for PHQ8_Binary (n_sessions=%d)" % seg.session_id.nunique())
    y, g = seg.phq8_binary.values, seg.session_id.values
    for rel in LAYERED:
        aucs = []
        n_layers = len([c for c in pq.read_schema(root / rel).names if c.startswith("mean_")])
        for k in range(n_layers):
            _, X = read_layer(root / rel, "mean", k)
            aucs.append(loso_auc(X, y, g))
        print(f"  {rel:32s} " + " ".join(f"{a:.2f}" for a in aucs))
    X = eg.drop(columns="segment_id").values
    print(f"  {'egemaps':32s} {loso_auc(X, y, g):.2f}")

    print("\nFAILED" if check.failed else "\nall checks passed")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
