"""End-to-end pipeline: download -> segment -> extract -> write (and optionally push) the HF dataset.

    python scripts/run_pipeline.py --config configs/sample.yaml [--push]
    python scripts/run_pipeline.py --config configs/full.yaml --push

Sessions are processed one at a time (the next download is prefetched in the background),
so memory and disk stay flat however many sessions there are. Every finished session
leaves a marker in <output.dir>/cache/done/, and a rerun skips those, so an interrupted
run resumes where it stopped.

With ``output.frames_upload_every: N`` and ``--push``, frame-level files are uploaded to
the HF repo every N sessions and deleted locally. Otherwise they stay in the output
folder and are uploaded with everything else at the end.

``--shard i/n`` processes only every n-th session (offset i), for parallel runs that share
an output repo; run once more without --shard (and with --assemble-only) to build the
final tables after all shards have finished and synced their caches.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
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
SEG_COLS = ["segment_id", "session_id", "turn_idx", "sub_idx", "start_s", "end_s", "dur_s", "text",
            "merged_from", "text_split_approx", "split", "phq8_score", "phq8_binary", "gender"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_revisions(cfg: dict) -> dict:
    api = HfApi()
    out = {}
    for group in ("models", "text_models"):
        for key, m in cfg[group].items():
            if not isinstance(m, dict):
                continue
            rev = m["revision"]
            if not re.fullmatch(r"[0-9a-f]{40}", rev):
                rev = api.model_info(m["id"], revision=rev).sha
            out[key] = {"id": m["id"], "revision": rev}
    return out


def load_audio(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == 16000, f"{path}: expected 16 kHz, got {sr}"
    return wav


def fetch_with_retry(base_url: str, data_dir: Path, s: int, tries: int = 4) -> None:
    for k in range(tries):
        try:
            daicwoz.fetch_session(base_url, data_dir, s)
            return
        except Exception as e:  # network hiccups
            if k == tries - 1:
                raise
            log(f"download {s} failed ({e!r}); retrying")
            time.sleep(10 * (k + 1))


class Extractors:
    """All models loaded once."""

    def __init__(self, revs: dict):
        from speechlab.extract_other import EGeMAPS, RobertaLayers, SentenceEmb
        from speechlab.extract_ssl import SSLExtractor

        self.ssl = {k: SSLExtractor(kind, revs[k]["id"], revs[k]["revision"]) for k, kind in SSL_KINDS.items()}
        self.egemaps = EGeMAPS()
        self.roberta = RobertaLayers(revs["roberta_base"]["id"], revs["roberta_base"]["revision"])
        self.mpnet = SentenceEmb(revs["mpnet"]["id"], revs["mpnet"]["revision"])


def process_session(s: int, cfg: dict, ex: Extractors, labels: pd.DataFrame, data_dir: Path,
                    cache: Path, hf_dir: Path, revs: dict) -> int:
    audio = load_audio(daicwoz.audio_path(data_dir, s))
    segs, st = segment_session(s, daicwoz.load_transcript(data_dir, s), audio, cfg["segmentation"])
    lab = labels.loc[s]
    rows = []
    for seg in segs:
        d = asdict(seg)
        d.update(segment_id=seg.segment_id, dur_s=round(seg.dur_s, 3), split=lab["split"],
                 phq8_score=int(lab["phq8_score"]), phq8_binary=int(lab["phq8_binary"]), gender=int(lab["gender"]))
        rows.append(d)
    (cache / "segments").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SEG_COLS).to_parquet(cache / "segments" / f"{s}.parquet", index=False)

    frame_layers = cfg["models"]["frame_layers"]
    timing = {}
    for key, ext in ex.ssl.items():
        t0 = time.time()
        means, stds, frames = [], [], {}
        for seg in segs:
            hs = ext.hidden_states(slice_audio(audio, seg))
            means.append(hs.mean(dim=1).numpy())
            stds.append(hs.std(dim=1, unbiased=False).numpy())
            for k in frame_layers:
                frames[f"{seg.segment_id}/layer_{k:02d}"] = hs[k].numpy().astype(np.float16)
        (cache / key).mkdir(parents=True, exist_ok=True)
        np.savez(cache / key / f"{s}_pooled.npz", mean=np.stack(means), std=np.stack(stds))
        fdir = hf_dir / key / "frames"
        fdir.mkdir(parents=True, exist_ok=True)
        save_file(frames, fdir / f"{s}.safetensors", metadata={
            "model": revs[key]["id"], "revision": revs[key]["revision"],
            "frame_rate_hz": "50", "layers": json.dumps(frame_layers)})
        timing[key] = round(time.time() - t0, 1)

    t0 = time.time()
    eg = pd.DataFrame([ex.egemaps(slice_audio(audio, seg)) for seg in segs])
    eg.insert(0, "segment_id", [seg.segment_id for seg in segs])
    (cache / "egemaps").mkdir(parents=True, exist_ok=True)
    eg.to_parquet(cache / "egemaps" / f"{s}.parquet", index=False)
    texts = [seg.text for seg in segs]
    (cache / "text").mkdir(parents=True, exist_ok=True)
    np.save(cache / "text" / f"{s}_roberta.npy", ex.roberta(texts))
    np.save(cache / "text" / f"{s}_mpnet.npy", ex.mpnet(texts))
    timing["egemaps_text"] = round(time.time() - t0, 1)

    st["timing_s"] = timing
    (cache / "done").mkdir(parents=True, exist_ok=True)
    (cache / "done" / f"{s}.json").write_text(json.dumps(st))  # written last = session complete
    return len(segs)


def ensure_private_repo(api: HfApi, repo: str, private: bool) -> None:
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    assert api.repo_info(repo, repo_type="dataset").private, f"{repo} is not private; refusing to upload"


def upload_frames(api: HfApi, repo: str, hf_dir: Path, ledger: Path) -> None:
    files = sorted(hf_dir.glob("*/frames/*.safetensors"))
    if not files:
        return
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(hf_dir),
                      allow_patterns=["*/frames/*.safetensors"],
                      commit_message=f"Frame-level outputs for {len(files) // 2} sessions")
    with ledger.open("a") as f:
        for p in files:
            f.write(str(p.relative_to(hf_dir)) + "\n")
            p.unlink()
    log(f"uploaded {len(files)} frame files")


def assemble(cfg: dict, cfg_text: str, sessions: list[int], cache: Path, hf_dir: Path, revs: dict) -> int:
    seg = pd.concat([pd.read_parquet(cache / "segments" / f"{s}.parquet") for s in sessions], ignore_index=True)
    ids = seg.segment_id.tolist()
    hf_dir.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(hf_dir / "segments.parquet", index=False)
    for key in SSL_KINDS:
        npz = [np.load(cache / key / f"{s}_pooled.npz") for s in sessions]
        store.write_layered(hf_dir / key / "pooled.parquet", ids,
                            {"mean": np.concatenate([z["mean"] for z in npz]),
                             "std": np.concatenate([z["std"] for z in npz])})
    (hf_dir / "egemaps").mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_parquet(cache / "egemaps" / f"{s}.parquet") for s in sessions], ignore_index=True) \
        .to_parquet(hf_dir / "egemaps" / "functionals.parquet", index=False)
    store.write_layered(hf_dir / "text" / "roberta_base.parquet", ids,
                        {"mean": np.concatenate([np.load(cache / "text" / f"{s}_roberta.npy") for s in sessions])})
    store.write_vectors(hf_dir / "text" / "mpnet.parquet", ids, "emb",
                        np.concatenate([np.load(cache / "text" / f"{s}_mpnet.npy") for s in sessions]),
                        dtype=np.float32)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    stats = {str(s): json.loads((cache / "done" / f"{s}.json").read_text()) for s in sessions}
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": commit + ("-dirty" if dirty else ""),
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "config": cfg,
        "models": revs,
        "libraries": {p: version(p) for p in ["torch", "transformers", "opensmile", "sentence-transformers",
                                              "librosa", "numpy", "pyarrow"]},
        "sessions": sessions,
        "segmentation_stats": stats,
        "n_segments": len(ids),
    }
    (hf_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    by_split = seg.groupby("split").session_id.nunique().to_dict()
    n_dep = int(seg.groupby("session_id").phq8_binary.first().sum())
    summary = (f"It covers {len(ids)} participant segments from {len(sessions)} sessions "
               f"(train {by_split.get('train', 0)}, dev {by_split.get('dev', 0)}, test {by_split.get('test', 0)}; "
               f"{n_dep} with PHQ-8 ≥ 10).")
    if len(sessions) <= 10:
        summary += (f" Sessions: {', '.join(map(str, sessions))}. This is a small test batch for exercising the "
                    "pipeline and probe code, far too small for meaningful results.")
    card = (ROOT / "scripts" / "dataset_card.md").read_text()
    for k, v in {"{{SUMMARY}}": summary, "{{REPO}}": cfg["output"]["hf_repo"],
                 "{{TITLE}}": "DAIC-WOZ segment embeddings" + (" (sample)" if len(sessions) <= 10 else "")}.items():
        card = card.replace(k, v)
    (hf_dir / "README.md").write_text(card)
    return len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/sample.yaml"))
    ap.add_argument("--push", action="store_true", help="upload to the HF repo")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shard", default=None, help="i/n: process only sessions[i::n]")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg_text = Path(args.config).read_text()
    cfg = yaml.safe_load(cfg_text)
    dcfg, ocfg = cfg["dataset"], cfg["output"]
    data_dir = ROOT / dcfg["data_dir"]
    out_dir = ROOT / ocfg["dir"]
    hf_dir, cache = out_dir / "hf", out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    base_url = dcfg["base_url"]

    daicwoz.fetch_meta(base_url, data_dir)
    labels = daicwoz.load_labels(data_dir)
    sessions = sorted(labels.index) if dcfg["sessions"] == "all" else list(dcfg["sessions"])
    sessions = [int(s) for s in sessions if s not in set(dcfg["exclude"])]
    todo = sessions
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        todo = sessions[i::n]
    revs = resolve_revisions(cfg)
    log(f"{len(sessions)} sessions in config, {len(todo)} in this run; revisions {json.dumps(revs)}")

    api = HfApi() if args.push else None
    repo = ocfg["hf_repo"]
    every = ocfg.get("frames_upload_every")
    ledger = cache / "uploaded_frames.txt"
    if api:
        ensure_private_repo(api, repo, ocfg["private"])

    pending = [s for s in todo if not (cache / "done" / f"{s}.json").exists()]
    log(f"{len(todo) - len(pending)} already done, {len(pending)} to process")
    if pending and not args.assemble_only:
        ex = Extractors(revs)
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(fetch_with_retry, base_url, data_dir, pending[0])
        t_start, n_segs = time.time(), 0
        for j, s in enumerate(pending):
            try:
                fut.result()
                fetch_err = None
            except zipfile.BadZipFile as e:  # the server's copy is broken: record it and move on
                fetch_err = e
            if j + 1 < len(pending):
                fut = pool.submit(fetch_with_retry, base_url, data_dir, pending[j + 1])
            if fetch_err is not None:
                (cache / "failed").mkdir(exist_ok=True)
                (cache / "failed" / f"{s}.json").write_text(json.dumps({"reason": f"download: {fetch_err!r}"}))
                log(f"session {s} skipped: source zip unreadable ({fetch_err!r})")
                continue
            t0 = time.time()
            n = process_session(s, cfg, ex, labels, data_dir, cache, hf_dir, revs)
            n_segs += n
            if not dcfg.get("keep_wav", True):
                daicwoz.audio_path(data_dir, s).unlink(missing_ok=True)
            el = time.time() - t_start
            eta = el / (j + 1) * (len(pending) - j - 1)
            log(f"session {s} ({j + 1}/{len(pending)}): {n} segments in {time.time() - t0:.0f}s; "
                f"elapsed {el / 3600:.2f}h, eta {eta / 3600:.2f}h")
            if api and every and (j + 1) % every == 0:
                upload_frames(api, repo, hf_dir, ledger)
        pool.shutdown()
        if api and every:
            upload_frames(api, repo, hf_dir, ledger)

    if args.shard:
        log("shard finished; run with --assemble-only once all shards are done")
        return
    failed = {int(p.stem): json.loads(p.read_text())["reason"] for p in sorted((cache / "failed").glob("*.json"))
              if not (cache / "done" / p.name).exists()}
    if failed:
        log(f"skipping {len(failed)} sessions whose source data is broken: {sorted(failed)}")
        sessions = [s for s in sessions if s not in failed]
        cfg["failed_sessions"] = {str(k): v for k, v in failed.items()}
    missing = [s for s in sessions if not (cache / "done" / f"{s}.json").exists()]
    if missing:
        log(f"not assembling: {len(missing)} sessions unfinished, e.g. {missing[:5]}")
        sys.exit(1)
    n = assemble(cfg, cfg_text, sessions, cache, hf_dir, revs)
    log(f"assembled {hf_dir} ({n} segments, {len(sessions)} sessions)")
    if api:
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(hf_dir),
                          commit_message=f"Embeddings for {len(sessions)} sessions")
        log(f"pushed to https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
