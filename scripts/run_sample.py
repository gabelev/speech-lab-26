"""End-to-end sample run: download -> segment -> extract -> write HF dataset folder.

    python scripts/run_sample.py --config configs/sample.yaml [--push]

Per-session results are cached under <output.dir>/cache so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from huggingface_hub import HfApi
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speechlab import daicwoz, store  # noqa: E402
from speechlab.segment import Segment, segment_session, slice_audio  # noqa: E402

SSL_KINDS = {"whisper_small": "whisper", "hubert_base": "hubert"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_revisions(cfg: dict) -> dict:
    api = HfApi()
    out = {}
    for group in ("models", "text_models"):
        for key, m in cfg[group].items():
            if not isinstance(m, dict):
                continue
            out[key] = {"id": m["id"], "revision": api.model_info(m["id"], revision=m["revision"]).sha}
    return out


def load_audio(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == 16000, f"{path}: expected 16 kHz, got {sr}"
    return wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/sample.yaml"))
    ap.add_argument("--push", action="store_true", help="upload the output folder to the HF repo")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg_text = Path(args.config).read_text()
    cfg = yaml.safe_load(cfg_text)
    data_dir = ROOT / cfg["dataset"]["data_dir"]
    out_dir = ROOT / cfg["output"]["dir"]
    hf_dir, cache = out_dir / "hf", out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    base_url = cfg["dataset"]["base_url"]
    sessions = [s for s in cfg["dataset"]["sessions"] if s not in set(cfg["dataset"]["exclude"])]

    revs = resolve_revisions(cfg)
    log(f"model revisions: {json.dumps(revs)}")

    # 1. data + segmentation -------------------------------------------------
    daicwoz.fetch_meta(base_url, data_dir)
    labels = daicwoz.load_labels(data_dir)
    all_segs: dict[int, list[Segment]] = {}
    seg_stats = {}
    audio = {}
    for s in sessions:
        daicwoz.fetch_session(base_url, data_dir, s)
        audio[s] = load_audio(daicwoz.audio_path(data_dir, s))
        segs, st = segment_session(s, daicwoz.load_transcript(data_dir, s), audio[s], cfg["segmentation"])
        all_segs[s], seg_stats[s] = segs, st
        log(f"session {s}: {st['segments']} segments, {st['speech_s']:.0f}s speech, "
            f"{st['rows_merged']} rows merged, {len(st['rows_dropped'])} dropped, {st['turns_split']} split")

    # 2. SSL models (cached per session) ----------------------------------------
    from speechlab.extract_ssl import SSLExtractor, pool

    frame_layers = cfg["models"]["frame_layers"]
    for key, kind in SSL_KINDS.items():
        ext = None
        for s in sessions:
            cpath = cache / key / f"{s}_pooled.npz"
            fpath = hf_dir / key / "frames" / f"{s}.safetensors"
            if cpath.exists() and fpath.exists():
                continue
            ext = ext or SSLExtractor(kind, revs[key]["id"], revs[key]["revision"])
            t0 = time.time()
            means, stds, frames = [], [], {}
            for seg in all_segs[s]:
                hs = ext.hidden_states(slice_audio(audio[s], seg))
                m, sd = pool(hs)
                means.append(m)
                stds.append(sd)
                for k in frame_layers:
                    frames[f"{seg.segment_id}/layer_{k:02d}"] = hs[k].numpy().astype(np.float16)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cpath, mean=np.stack(means), std=np.stack(stds))
            fpath.parent.mkdir(parents=True, exist_ok=True)
            save_file(frames, fpath, metadata={"model": revs[key]["id"], "revision": revs[key]["revision"],
                                               "frame_rate_hz": "50", "layers": json.dumps(frame_layers)})
            log(f"{key} session {s}: {len(all_segs[s])} segments in {time.time() - t0:.0f}s")
        del ext

    # 3. eGeMAPS -------------------------------------------------------------
    from speechlab.extract_other import EGeMAPS, RobertaLayers, SentenceEmb

    eg_path = cache / "egemaps.parquet"
    if not eg_path.exists():
        eg = EGeMAPS()
        rows = []
        for s in sessions:
            for seg in all_segs[s]:
                r = eg(slice_audio(audio[s], seg))
                r.name = seg.segment_id
                rows.append(r)
        pd.DataFrame(rows).rename_axis("segment_id").reset_index().to_parquet(eg_path)
        log("egemaps done")

    # 4. text models -----------------------------------------------------------
    ids = [seg.segment_id for s in sessions for seg in all_segs[s]]
    texts = [seg.text for s in sessions for seg in all_segs[s]]
    rb_path, mp_path = cache / "roberta_base.npy", cache / "mpnet.npy"
    if not rb_path.exists():
        np.save(rb_path, RobertaLayers(revs["roberta_base"]["id"], revs["roberta_base"]["revision"])(texts))
        log("roberta done")
    if not mp_path.exists():
        np.save(mp_path, SentenceEmb(revs["mpnet"]["id"], revs["mpnet"]["revision"])(texts))
        log("mpnet done")

    # 5. assemble HF folder ------------------------------------------------------
    seg_rows = []
    for s in sessions:
        lab = labels.loc[s]
        for seg in all_segs[s]:
            d = asdict(seg)
            d.update(segment_id=seg.segment_id, dur_s=round(seg.dur_s, 3), split=lab["split"],
                     phq8_score=int(lab["phq8_score"]), phq8_binary=int(lab["phq8_binary"]),
                     gender=int(lab["gender"]))
            seg_rows.append(d)
    seg_df = pd.DataFrame(seg_rows)[["segment_id", "session_id", "turn_idx", "sub_idx", "start_s", "end_s",
                                     "dur_s", "text", "merged_from", "text_split_approx", "split",
                                     "phq8_score", "phq8_binary", "gender"]]
    seg_df.to_parquet(hf_dir / "segments.parquet", index=False)

    for key in SSL_KINDS:
        npz = [np.load(cache / key / f"{s}_pooled.npz") for s in sessions]
        store.write_layered(hf_dir / key / "pooled.parquet", ids,
                            {"mean": np.concatenate([z["mean"] for z in npz]),
                             "std": np.concatenate([z["std"] for z in npz])})
    eg_df = pd.read_parquet(eg_path)
    (hf_dir / "egemaps").mkdir(parents=True, exist_ok=True)
    eg_df.to_parquet(hf_dir / "egemaps" / "functionals.parquet", index=False)
    store.write_layered(hf_dir / "text" / "roberta_base.parquet", ids, {"mean": np.load(rb_path)})
    store.write_vectors(hf_dir / "text" / "mpnet.parquet", ids, "emb", np.load(mp_path), dtype=np.float32)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": commit + ("-dirty" if dirty else ""),
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "config": cfg,
        "models": revs,
        "libraries": {p: version(p) for p in ["torch", "transformers", "opensmile", "sentence-transformers",
                                              "librosa", "numpy", "pyarrow"]},
        "sessions": sessions,
        "segmentation_stats": {str(k): v for k, v in seg_stats.items()},
        "n_segments": len(ids),
    }
    (hf_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    card = (ROOT / "scripts" / "dataset_card.md").read_text()
    (hf_dir / "README.md").write_text(card.replace("{{N_SEGMENTS}}", str(len(ids)))
                                      .replace("{{SESSIONS}}", ", ".join(map(str, sessions))))
    log(f"wrote {hf_dir} ({len(ids)} segments)")

    if args.push:
        api = HfApi()
        repo = cfg["output"]["hf_repo"]
        api.create_repo(repo, repo_type="dataset", private=cfg["output"]["private"], exist_ok=True)
        info = api.repo_info(repo, repo_type="dataset")
        assert info.private, f"{repo} is not private; refusing to upload EULA-covered data"
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(hf_dir),
                          commit_message=f"Sample embeddings ({len(sessions)} sessions) from {manifest['code_commit']}")
        log(f"pushed to https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
