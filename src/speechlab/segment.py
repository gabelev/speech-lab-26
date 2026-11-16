"""Participant-turn segmentation shared by every feature extractor.

Rules (see configs/*.yaml for values):
1. Keep Participant rows; drop Ellie rows and anything before Ellie's first line
   (pre-interview setup audio).
2. A turn shorter than ``min_dur_s`` is merged into the previous participant turn if
   no Ellie row sits between them and the gap is <= ``merge_max_gap_s``; otherwise into
   the next one under the same conditions; otherwise dropped. Merged segments still
   shorter than ``min_dur_s`` are dropped.
3. Segments longer than ``max_dur_s`` are split at the lowest-RMS 10 ms frame within
   ``split_search_s`` before the limit. Transcript text is divided between the pieces
   in proportion to time (``text_split_approx`` = True), since word timings are not
   available.

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
    start_s: float
    end_s: float
    text: str
    merged_from: list[int] = field(default_factory=list)  # transcript row indices
    text_split_approx: bool = False

    @property
    def segment_id(self) -> str:
        return f"{self.session_id}_{self.turn_idx:04d}_{self.sub_idx}"

    @property
    def dur_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class _Turn:
    start: float
    end: float
    text: str
    rows: list[int]
    pos: int  # position in full transcript (to detect intervening Ellie rows)
    last_pos: int


def _participant_turns(tr: pd.DataFrame) -> list[_Turn]:
    is_ellie = tr["speaker"].str.lower() == "ellie"
    first_ellie = tr.loc[is_ellie, "start_time"].min() if is_ellie.any() else -np.inf
    turns = []
    for pos, (i, r) in enumerate(tr.iterrows()):
        if r["speaker"].lower() != "participant" or r["start_time"] < first_ellie:
            continue
        if r["stop_time"] <= r["start_time"]:
            continue
        turns.append(_Turn(float(r["start_time"]), float(r["stop_time"]), r["value"], [int(i)], pos, pos))
    return turns


def merge_short_turns(tr: pd.DataFrame, min_dur_s: float, max_gap_s: float) -> tuple[list[_Turn], list[int]]:
    """Returns (kept turns, dropped transcript row indices)."""
    turns = _participant_turns(tr)
    out: list[_Turn] = []
    dropped: list[int] = []
    pending: _Turn | None = None  # short turn waiting to merge into the next one

    def adjacent(a: _Turn, b: _Turn) -> bool:
        # consecutive participant rows (no Ellie in between) with a small gap
        return b.pos == a.last_pos + 1 and (b.start - a.end) <= max_gap_s

    for t in turns:
        if pending is not None:
            if adjacent(pending, t):
                t = _Turn(pending.start, t.end, f"{pending.text} {t.text}".strip(),
                          pending.rows + t.rows, pending.pos, t.last_pos)
            else:
                dropped.extend(pending.rows)
            pending = None
        if t.end - t.start >= min_dur_s:
            out.append(t)
            continue
        if out and adjacent(out[-1], t):
            p = out[-1]
            out[-1] = _Turn(p.start, t.end, f"{p.text} {t.text}".strip(), p.rows + t.rows, p.pos, t.last_pos)
        else:
            pending = t
    if pending is not None:
        dropped.extend(pending.rows)
    kept = []
    for t in out:
        if t.end - t.start >= min_dur_s:
            kept.append(t)
        else:
            dropped.extend(t.rows)
    return kept, sorted(dropped)


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


def segment_session(session_id: int, tr: pd.DataFrame, audio: np.ndarray, cfg: dict) -> tuple[list[Segment], dict]:
    audio_dur = len(audio) / SR
    kept, dropped = merge_short_turns(tr, cfg["min_dur_s"], cfg["merge_max_gap_s"])
    segs: list[Segment] = []
    for turn_idx, t in enumerate(kept):
        start, end = t.start, min(t.end, audio_dur)
        cuts = _split_points(audio, start, end, cfg["max_dur_s"], cfg["split_search_s"])
        edges = [start, *cuts, end]
        bounds = list(zip(edges[:-1], edges[1:]))
        texts = _split_text(t.text, bounds) if cuts else [t.text]
        for sub_idx, ((s, e), txt) in enumerate(zip(bounds, texts)):
            segs.append(Segment(session_id, turn_idx, sub_idx, round(s, 3), round(e, 3), txt,
                                list(t.rows), bool(cuts)))
    stats = {
        "participant_rows": int((tr["speaker"].str.lower() == "participant").sum()),
        "turns_kept": len(kept),
        "rows_merged": int(sum(len(t.rows) - 1 for t in kept)),
        "rows_dropped": dropped,
        "turns_split": int(sum(1 for s in segs if s.sub_idx == 1)),
        "segments": len(segs),
        "speech_s": round(sum(s.dur_s for s in segs), 2),
    }
    return segs, stats


def slice_audio(audio: np.ndarray, seg: Segment) -> np.ndarray:
    return audio[int(round(seg.start_s * SR)): int(round(seg.end_s * SR))]
