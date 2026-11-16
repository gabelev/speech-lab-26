"""DAIC-WOZ download, file access and labels."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

SPLIT_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "full_test_split.csv",  # the AVEC2017 test file has no labels; this one does
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def fetch_meta(base_url: str, data_dir: Path) -> None:
    for name in SPLIT_FILES.values():
        dest = data_dir / "meta" / name
        if not dest.exists():
            _download(base_url + name, dest)


def fetch_session(base_url: str, data_dir: Path, session: int) -> Path:
    """Download <id>_P.zip, keep only the audio and transcript, delete the zip."""
    sdir = data_dir / "sessions" / str(session)
    wanted = [f"{session}_AUDIO.wav", f"{session}_TRANSCRIPT.csv"]
    if all((sdir / w).exists() for w in wanted):
        return sdir
    zpath = data_dir / "sessions" / f"{session}_P.zip"
    _download(f"{base_url}{session}_P.zip", zpath)
    with zipfile.ZipFile(zpath) as z:
        for w in wanted:
            z.extract(w, sdir)
    zpath.unlink()
    return sdir


def audio_path(data_dir: Path, session: int) -> Path:
    return data_dir / "sessions" / str(session) / f"{session}_AUDIO.wav"


def load_transcript(data_dir: Path, session: int) -> pd.DataFrame:
    p = data_dir / "sessions" / str(session) / f"{session}_TRANSCRIPT.csv"
    df = pd.read_csv(p, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    df["speaker"] = df["speaker"].astype(str).str.strip()
    df["value"] = df["value"].fillna("").astype(str).str.strip()
    return df[["start_time", "stop_time", "speaker", "value"]]


def load_labels(data_dir: Path) -> pd.DataFrame:
    """One row per participant: split, phq8_score, phq8_binary, gender (0/1 as in source)."""
    frames = []
    for split, name in SPLIT_FILES.items():
        df = pd.read_csv(data_dir / "meta" / name)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(
            columns={
                "Participant_ID": "session_id",
                "PHQ8_Binary": "phq8_binary",
                "PHQ_Binary": "phq8_binary",
                "PHQ8_Score": "phq8_score",
                "PHQ_Score": "phq8_score",
                "Gender": "gender",
            }
        )
        df["split"] = split
        frames.append(df[["session_id", "split", "phq8_score", "phq8_binary", "gender"]])
    return pd.concat(frames, ignore_index=True).set_index("session_id")
