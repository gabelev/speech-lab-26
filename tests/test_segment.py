import numpy as np
import pandas as pd

from speechlab.extract_ssl import whisper_valid_frames
from speechlab.segment import SR, merge_short_turns, segment_session

CFG = {"min_dur_s": 1.0, "merge_max_gap_s": 0.5, "max_dur_s": 30.0, "split_search_s": 3.0}


def tr(rows):
    return pd.DataFrame(rows, columns=["start_time", "stop_time", "speaker", "value"])


def test_drops_ellie_and_pre_interview():
    t = tr([(0.0, 5.0, "Participant", "setup chatter"),
            (10.0, 12.0, "Ellie", "hi"),
            (13.0, 16.0, "Participant", "hello there")])
    kept, dropped = merge_short_turns(t, 1.0, 0.5)
    assert [(k.start, k.end, k.text) for k in kept] == [(13.0, 16.0, "hello there")]
    assert dropped == []


def test_short_turn_merges_backward_then_forward_then_drops():
    t = tr([(0.0, 1.0, "Ellie", "q"),
            (2.0, 4.0, "Participant", "long one"),
            (4.2, 4.6, "Participant", "uh"),        # merges back into previous (gap 0.2)
            (6.0, 7.0, "Ellie", "q2"),
            (8.0, 8.5, "Participant", "so"),        # merges forward (gap 0.3)
            (8.8, 10.0, "Participant", "anyway yes"),
            (11.0, 12.0, "Ellie", "q3"),
            (13.0, 13.4, "Participant", "no")])     # Ellie before and nothing after -> dropped
    kept, dropped = merge_short_turns(t, 1.0, 0.5)
    assert [(k.start, k.end, k.text, k.rows) for k in kept] == [
        (2.0, 4.6, "long one uh", [1, 2]),
        (8.0, 10.0, "so anyway yes", [4, 5]),
    ]
    assert dropped == [7]


def test_no_merge_across_ellie():
    t = tr([(0.0, 0.5, "Ellie", "q"),
            (1.0, 3.0, "Participant", "answer"),
            (3.1, 3.3, "Ellie", "mhm"),
            (3.4, 3.8, "Participant", "ok")])
    kept, dropped = merge_short_turns(t, 1.0, 0.5)
    assert len(kept) == 1 and dropped == [3]


def test_long_turn_split_at_pause_and_ids_stable():
    dur = 70.0
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.1, int(dur * SR)).astype(np.float32)
    audio[int(28.0 * SR): int(28.2 * SR)] = 0.0  # pause inside first search window [27, 30]
    t = tr([(0.0, 0.5, "Ellie", "q"), (1.0, 69.0, "Participant", " ".join(["w"] * 68))])
    segs, stats = segment_session(300, t, audio, CFG)
    assert all(s.dur_s <= 30.0 + 1e-6 for s in segs)
    assert 28.0 <= segs[0].end_s <= 28.2
    assert [s.segment_id for s in segs] == [f"300_0000_{i}" for i in range(len(segs))]
    assert segs[0].start_s == 1.0 and segs[-1].end_s == 69.0
    assert sum(len(s.text.split()) for s in segs) == 68
    assert all(s.text_split_approx for s in segs)
    assert stats["turns_split"] == 1


def test_whisper_frames():
    assert whisper_valid_frames(SR) == 50
    assert whisper_valid_frames(SR + 1) == 51
    assert whisper_valid_frames(SR * 30) == 1500
    assert whisper_valid_frames(SR * 31) == 1500
