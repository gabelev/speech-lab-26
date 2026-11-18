"""Participant-turn segmentation shared by every feature extractor (dataset v2).

Rules (see configs/*.yaml for values):
1. Keep Participant rows; drop Ellie rows. Participant rows before Ellie's first line
   (pre-interview setup audio) and rows with zero or negative length are the only
   participant rows that are dropped.
2. Every other participant row ends up in exactly one segment. A turn shorter than
   ``min_dur_s`` is merged into the participant turn nearest in time (previous or next;
   ties go to the previous one), repeatedly, until it is long enough:
   - if no Ellie row sits between the two, they become one contiguous span (the pause
     between them is kept);
   - if an Ellie row sits between them, the short turn's audio is *spliced* on as a
     separate interval, so the segment stays participant-only.
   A session whose only participant turn is short keeps it as a short segment.
3. Segments whose speech exceeds ``max_dur_s`` are split at the lowest-RMS 10 ms frame
   within ``split_search_s`` before the limit, measured on the concatenated (spliced)
   audio. Transcript text is divided between the pieces in proportion to time
   (``text_split_approx`` = True), since word timings are not available.

A segment is a list of ``intervals`` (start, end) in session time; its audio is the
concatenation of those slices and ``dur_s`` is their total length.
``segment_id = f"{session}_{turn_idx:04d}_{sub_idx}"`` is the join key for all tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SR = 16000


@dataclass
class Segment:
    session_id: int
    turn_idx: int
    sub_idx: int
    intervals: list[list[float]]
    text: str
    merged_from: list[int] = field(default_factory=list)  # transcript row indices
    text_split_approx: bool = False
    n_short_merged: int = 0  # transcript rows shorter than min_dur_s folded into this turn
    spliced: bool = False    # audio joins pieces with an Ellie line cut out between them

    @property
    def segment_id(self) -> str:
        return f"{self.session_id}_{self.turn_idx:04d}_{self.sub_idx}"

    @property
    def start_s(self) -> float:
        return self.intervals[0][0]

    @property
    def end_s(self) -> float:
        return self.intervals[-1][1]

    @property
    def dur_s(self) -> float:
        return float(sum(e - s for s, e in self.intervals))


@dataclass
class _Turn:
    intervals: list[tuple[float, float]]
    text: str
    rows: list[int]
    first_pos: int  # position in the full transcript (to detect intervening Ellie rows)
    last_pos: int
    n_short: int = 0
    spliced: bool = False

    @property
    def dur(self) -> float:
        return sum(e - s for s, e in self.intervals)


def _participant_turns(tr: pd.DataFrame) -> tuple[list[_Turn], dict]:
    is_ellie = tr["speaker"].str.lower() == "ellie"
    first_ellie = tr.loc[is_ellie, "start_time"].min() if is_ellie.any() else -np.inf
    turns, pre, nonpos = [], [], []
    for pos, (i, r) in enumerate(tr.iterrows()):
        if r["speaker"].lower() != "participant":
            continue
        if r["start_time"] < first_ellie:
            pre.append(int(i))
            continue
        if r["stop_time"] <= r["start_time"]:
            nonpos.append(int(i))
            continue
        turns.append(_Turn([(float(r["start_time"]), float(r["stop_time"]))], r["value"], [int(i)], pos, pos))
    return turns, {"rows_pre_interview": pre, "rows_nonpositive": nonpos}


def _join(a: _Turn, b: _Turn) -> _Turn:
    """a precedes b in time."""
    if b.first_pos == a.last_pos + 1:  # no Ellie row between: one contiguous span
        iv = a.intervals[:-1] + [(a.intervals[-1][0], b.intervals[0][1])] + b.intervals[1:]
        spliced = a.spliced or b.spliced
    else:  # splice, leaving Ellie's audio out
        iv = a.intervals + b.intervals
        spliced = True
    return _Turn(iv, f"{a.text} {b.text}".strip(), a.rows + b.rows, a.first_pos, b.last_pos,
                 a.n_short + b.n_short, spliced)


def merge_short_turns(tr: pd.DataFrame, min_dur_s: float) -> tuple[list[_Turn], dict]:
    """Fold every turn shorter than ``min_dur_s`` into its nearest participant neighbour.
    Returns (turns, info) where info lists the dropped rows by reason."""
    turns, info = _participant_turns(tr)
    for t in turns:
        t.n_short = int(t.dur < min_dur_s)
    while len(turns) > 1:
        k = next((i for i, t in enumerate(turns) if t.dur < min_dur_s), None)
        if k is None:
            break
        t = turns[k]
        gap_prev = t.intervals[0][0] - turns[k - 1].intervals[-1][1] if k > 0 else np.inf
        gap_next = turns[k + 1].intervals[0][0] - t.intervals[-1][1] if k + 1 < len(turns) else np.inf
        if gap_prev <= gap_next:
            turns[k - 1: k + 1] = [_join(turns[k - 1], t)]
        else:
            turns[k: k + 2] = [_join(t, turns[k + 1])]
    return turns, info


def _split_points(audio: np.ndarray, start: float, end: float, max_dur: float, search: float) -> list[float]:
    """Cut times so every piece is <= max_dur, each at the quietest 10 ms frame in the search window."""
    cuts = []
    cur = start
    hop = SR // 100
    while end - cur > max_dur:
        lo, hi = cur + max_dur - search, cur + max_dur
        a = audio[int(lo * SR): int(hi * SR)]
        n = len(a) // hop
        if n == 0:
            cut = hi
        else:
            rms = np.sqrt((a[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(axis=1))
            cut = lo + float(np.argmin(rms)) * hop / SR + 0.005
        cuts.append(cut)
        cur = cut
    return cuts


def _split_text(text: str, bounds: list[tuple[float, float]]) -> list[str]:
    words = text.split()
    total = bounds[-1][1] - bounds[0][0]
    pieces, used = [], 0
    for k, (s, e) in enumerate(bounds):
        if k == len(bounds) - 1:
            n = len(words) - used
        else:
            n = round(len(words) * (e - bounds[0][0]) / total) - used
        pieces.append(" ".join(words[used: used + n]))
        used += n
    return pieces


def _to_session_time(intervals: list[tuple[float, float]], v0: float, v1: float) -> list[tuple[float, float]]:
    """Map [v0, v1) on the concatenated timeline back to session-time intervals."""
    out, offset = [], 0.0
    for s, e in intervals:
        d = e - s
        lo, hi = max(v0, offset), min(v1, offset + d)
        if hi - lo > 1e-9:
            out.append((s + lo - offset, s + hi - offset))
        offset += d
    return out


def _concat(audio: np.ndarray, intervals) -> np.ndarray:
    return np.concatenate([audio[int(round(s * SR)): int(round(e * SR))] for s, e in intervals])


def segment_session(session_id: int, tr: pd.DataFrame, audio: np.ndarray, cfg: dict) -> tuple[list[Segment], dict]:
    audio_dur = len(audio) / SR
    turns, info = merge_short_turns(tr, cfg["min_dur_s"])
    segs: list[Segment] = []
    for turn_idx, t in enumerate(turns):
        iv = [(s, min(e, audio_dur)) for s, e in t.intervals if s < audio_dur]
        if not iv:
            continue
        total = sum(e - s for s, e in iv)
        cuts = _split_points(_concat(audio, iv), 0.0, total, cfg["max_dur_s"], cfg["split_search_s"])
        edges = [0.0, *cuts, total]
        vbounds = list(zip(edges[:-1], edges[1:]))
        texts = _split_text(t.text, vbounds) if cuts else [t.text]
        for sub_idx, ((v0, v1), txt) in enumerate(zip(vbounds, texts)):
            piece = [[round(s, 3), round(e, 3)] for s, e in _to_session_time(iv, v0, v1)]
            segs.append(Segment(session_id, turn_idx, sub_idx, piece, txt, list(t.rows), bool(cuts),
                                t.n_short, len(piece) > 1 and t.spliced))
    stats = {
        "participant_rows": int((tr["speaker"].str.lower() == "participant").sum()),
        "rows_pre_interview": info["rows_pre_interview"],
        "rows_nonpositive": info["rows_nonpositive"],
        "rows_dropped": info["rows_pre_interview"] + info["rows_nonpositive"],
        "rows_short": int(sum(t.n_short for t in turns)),
        "turns_kept": len(turns),
        "rows_merged": int(sum(len(t.rows) - 1 for t in turns)),
        "turns_spliced": int(sum(t.spliced for t in turns)),
        "turns_still_short": int(sum(t.dur < cfg["min_dur_s"] for t in turns)),
        "turns_split": int(sum(1 for s in segs if s.sub_idx == 1)),
        "segments": len(segs),
        "speech_s": round(sum(s.dur_s for s in segs), 2),
    }
    return segs, stats


def slice_audio(audio: np.ndarray, seg: Segment) -> np.ndarray:
    return _concat(audio, seg.intervals)
