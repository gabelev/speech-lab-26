import numpy as np
import pandas as pd

from speechlab.extract_ssl import whisper_valid_frames
from speechlab.segment import SR, merge_short_turns, segment_session, slice_audio

CFG = {"min_dur_s": 1.0, "max_dur_s": 30.0, "split_search_s": 3.0}


def tr(rows):
    return pd.DataFrame(rows, columns=["start_time", "stop_time", "speaker", "value"])


def test_drops_only_ellie_pre_interview_and_empty_rows():
    t = tr([(0.0, 5.0, "Participant", "setup chatter"),
            (10.0, 12.0, "Ellie", "hi"),
            (13.0, 16.0, "Participant", "hello there"),
            (16.5, 16.5, "Participant", "")])
    turns, info = merge_short_turns(t, 1.0)
    assert [(k.intervals, k.text) for k in turns] == [([(13.0, 16.0)], "hello there")]
    assert info == {"rows_pre_interview": [0], "rows_nonpositive": [3]}


def test_short_turns_merge_into_nearest_neighbour_span_or_splice():
    t = tr([(0.0, 1.0, "Ellie", "q"),
            (2.0, 4.0, "Participant", "long one"),
            (4.9, 5.3, "Participant", "uh"),        # 0.9 s after prev, 2.7 s before next -> span into prev
            (6.0, 7.0, "Ellie", "q2"),
            (8.0, 8.5, "Participant", "so"),        # 0.3 s before next -> span into next
            (8.8, 10.0, "Participant", "anyway yes"),
            (11.0, 12.0, "Ellie", "q3"),
            (13.0, 13.4, "Participant", "no")])     # Ellie on the left, nothing after -> spliced onto prev
    turns, _ = merge_short_turns(t, 1.0)
    got = [(k.intervals, k.text, k.rows, k.n_short, k.spliced) for k in turns]
    assert got == [
        ([(2.0, 5.3)], "long one uh", [1, 2], 1, False),
        ([(8.0, 10.0), (13.0, 13.4)], "so anyway yes no", [4, 5, 7], 2, True),
    ]
    rows = sorted(r for k in turns for r in k.rows)
    assert rows == [1, 2, 4, 5, 7]  # every participant row kept


def test_isolated_short_answer_goes_to_closer_side():
    t = tr([(0.0, 0.5, "Ellie", "q"),
            (1.0, 3.0, "Participant", "answer"),
            (3.1, 3.3, "Ellie", "mhm"),
            (3.4, 3.8, "Participant", "ok"),        # 0.4 s after prev, 3.2 s before next
            (5.0, 6.0, "Ellie", "q2"),
            (7.0, 9.0, "Participant", "second")])
    turns, _ = merge_short_turns(t, 1.0)
    assert [k.intervals for k in turns] == [[(1.0, 3.0), (3.4, 3.8)], [(7.0, 9.0)]]
    assert turns[0].spliced and not turns[1].spliced


def test_only_short_turn_is_kept():
    t = tr([(0.0, 1.0, "Ellie", "q"), (2.0, 2.4, "Participant", "no")])
    turns, _ = merge_short_turns(t, 1.0)
    assert len(turns) == 1 and turns[0].dur < 1.0


def test_spliced_audio_and_duration():
    rng = np.random.default_rng(1)
    audio = rng.normal(0, 0.1, 20 * SR).astype(np.float32)
    t = tr([(0.0, 0.5, "Ellie", "q"), (1.0, 3.0, "Participant", "answer"),
            (3.1, 3.3, "Ellie", "mhm"), (3.4, 3.8, "Participant", "ok")])
    segs, st = segment_session(300, t, audio, CFG)
    assert len(segs) == 1 and segs[0].spliced and segs[0].intervals == [[1.0, 3.0], [3.4, 3.8]]
    clip = slice_audio(audio, segs[0])
    assert len(clip) == round(segs[0].dur_s * SR)
    np.testing.assert_array_equal(clip[2 * SR:], audio[int(3.4 * SR): int(3.8 * SR)])
    assert st["rows_dropped"] == [] and st["rows_short"] == 1 and st["turns_spliced"] == 1


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


def test_split_on_spliced_timeline_maps_back_to_session_time():
    rng = np.random.default_rng(2)
    audio = rng.normal(0, 0.1, 60 * SR).astype(np.float32)
    # 29 s answer + Ellie + 0.5 s "yeah" spliced on -> 29.5 s of speech, then a 2 s pause-free tail
    t = tr([(0.0, 0.5, "Ellie", "q"), (1.0, 30.0, "Participant", "a " * 29),
            (30.2, 31.0, "Ellie", "mhm"), (31.2, 31.7, "Participant", "yeah"),
            (40.0, 41.0, "Ellie", "q2"), (42.0, 44.0, "Participant", "next")])
    segs, _ = segment_session(300, t, audio, {**CFG, "max_dur_s": 20.0})
    first = [s for s in segs if s.turn_idx == 0]
    assert len(first) == 2
    assert abs(sum(s.dur_s for s in first) - 29.5) < 1e-6
    assert first[-1].intervals[-1] == [31.2, 31.7] and first[-1].spliced and not first[0].spliced
    whole = np.concatenate([slice_audio(audio, s) for s in first])
    ref = np.concatenate([audio[1 * SR: 30 * SR], audio[int(31.2 * SR): int(31.7 * SR)]])
    assert abs(len(whole) - len(ref)) <= 2


def test_whisper_frames():
    assert whisper_valid_frames(SR) == 50
    assert whisper_valid_frames(SR + 1) == 51
    assert whisper_valid_frames(SR * 30) == 1500
    assert whisper_valid_frames(SR * 31) == 1500
