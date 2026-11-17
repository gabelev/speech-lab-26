"""Read the embedding dataset written by ``scripts/run_sample.py``.

Works on a local output folder or on the private HF repo (downloaded on first use and
cached by ``huggingface_hub``). Every table shares the ``segment_id`` key and is stored
in the same row order as ``segments.parquet``.

    from speechlab.load import EmbeddingDataset

    ds = EmbeddingDataset.from_hub("GenerationGenerative/daic-woz-embeddings-sample")
    seg = ds.segments()                                   # metadata DataFrame
    X = ds.pooled("hubert_base", layer=9)                 # [N, 768] float32 (mean)
    S = ds.pooled("whisper_small", layer=4, stat="std")   # [N, 768]
    A = ds.pooled_all_layers("whisper_small")             # [N, 13, 768]
    F = ds.frames("hubert_base", "303_0000_0", layer=6)   # [T, 768] at 50 Hz
    E = ds.egemaps()                                      # DataFrame, 88 columns
    T = ds.text("roberta_base", layer=12)                 # [N, 768]
    M = ds.text("mpnet")                                  # [N, 768]
    Xs, ys, sessions = ds.session_pool(X)                 # one row per session
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SPEECH_MODELS = ("whisper_small", "hubert_base")
TEXT_MODELS = ("roberta_base", "mpnet")
N_LAYERS = 13


def _vectors(path: Path, column: str) -> np.ndarray:
    col = pq.read_table(path, columns=[column]).column(0).combine_chunks()
    return col.flatten().to_numpy().reshape(len(col), col.type.list_size).astype(np.float32)


class EmbeddingDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / "segments.parquet").exists():
            raise FileNotFoundError(f"{self.root} does not look like an embedding dataset folder")

    @classmethod
    def from_hub(cls, repo_id: str = "GenerationGenerative/daic-woz-embeddings-sample", revision: str | None = None,
                 frames: bool = True) -> "EmbeddingDataset":
        """Download (or reuse the cache of) the HF dataset repo. Needs a token with read
        access to the private repo (``huggingface-cli login`` or ``HF_TOKEN``).
        Pass ``frames=False`` to skip the frame-level files (most of the download)."""
        from huggingface_hub import snapshot_download

        ignore = None if frames else ["*/frames/*"]
        return cls(snapshot_download(repo_id, repo_type="dataset", revision=revision, ignore_patterns=ignore))

    # ---- metadata ---------------------------------------------------------------
    @cached_property
    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text())

    def segments(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "segments.parquet")

    @cached_property
    def segment_ids(self) -> list[str]:
        return pq.read_table(self.root / "segments.parquet", columns=["segment_id"]).column(0).to_pylist()

    # ---- speech models ------------------------------------------------------------
    def pooled(self, model: str, layer: int, stat: str = "mean") -> np.ndarray:
        """[N, 768] float32 for one layer (0 = pre-transformer, 12 = last block)."""
        if model not in SPEECH_MODELS:
            raise ValueError(f"model must be one of {SPEECH_MODELS}")
        if stat not in ("mean", "std"):
            raise ValueError("stat must be 'mean' or 'std'")
        return _vectors(self.root / model / "pooled.parquet", f"{stat}_l{layer:02d}")

    def pooled_all_layers(self, model: str, stat: str = "mean") -> np.ndarray:
        """[N, 13, 768] float32."""
        return np.stack([self.pooled(model, k, stat) for k in range(N_LAYERS)], axis=1)

    def frame_layers(self) -> list[int]:
        return list(self.manifest["config"]["models"]["frame_layers"])

    def frames(self, model: str, segment_id: str, layer: int) -> np.ndarray:
        """[T, 768] float32 frame-level hidden states (50 frames/s) for one segment.
        Only the layers in ``frame_layers()`` are stored."""
        from safetensors import safe_open

        if layer not in self.frame_layers():
            raise ValueError(f"layer {layer} not stored at frame level; available: {self.frame_layers()}")
        session = segment_id.split("_")[0]
        with safe_open(self.root / model / "frames" / f"{session}.safetensors", framework="numpy") as f:
            return f.get_tensor(f"{segment_id}/layer_{layer:02d}").astype(np.float32)

    def session_frames(self, model: str, session_id: int, layer: int) -> dict[str, np.ndarray]:
        """All segments of one session: {segment_id: [T, 768]}."""
        from safetensors import safe_open

        out = {}
        suffix = f"/layer_{layer:02d}"
        with safe_open(self.root / model / "frames" / f"{session_id}.safetensors", framework="numpy") as f:
            for k in f.keys():
                if k.endswith(suffix):
                    out[k[: -len(suffix)]] = f.get_tensor(k).astype(np.float32)
        return out

    # ---- eGeMAPS and text ---------------------------------------------------------
    def egemaps(self) -> pd.DataFrame:
        """88 eGeMAPSv02 functionals, indexed by segment_id."""
        return pd.read_parquet(self.root / "egemaps" / "functionals.parquet").set_index("segment_id")

    def text(self, model: str, layer: int | None = None) -> np.ndarray:
        """roberta_base: needs ``layer`` (0-12) -> [N, 768]; mpnet: sentence embedding -> [N, 768]."""
        if model == "roberta_base":
            if layer is None:
                raise ValueError("roberta_base needs a layer (0-12)")
            return _vectors(self.root / "text" / "roberta_base.parquet", f"mean_l{layer:02d}")
        if model == "mpnet":
            return _vectors(self.root / "text" / "mpnet.parquet", "emb")
        raise ValueError(f"model must be one of {TEXT_MODELS}")

    # ---- helpers ----------------------------------------------------------------
    def session_pool(self, X: np.ndarray, weight_by_duration: bool = False
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average segment rows into one row per session.
        Returns (X_session [S, D], phq8_binary [S], session_ids [S])."""
        seg = self.segments()
        sessions = seg.session_id.unique()
        Xs, ys = [], []
        for s in sessions:
            m = (seg.session_id == s).values
            w = seg.dur_s.values[m] if weight_by_duration else None
            Xs.append(np.average(X[m], axis=0, weights=w))
            ys.append(seg.phq8_binary.values[m][0])
        return np.stack(Xs), np.asarray(ys), sessions
