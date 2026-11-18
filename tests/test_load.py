import json

import numpy as np
import pandas as pd
from safetensors.numpy import save_file

from speechlab import store
from speechlab.load import EmbeddingDataset


def _make(root, n=4):
    ids = [f"300_{i:04d}_0" for i in range(2)] + [f"301_{i:04d}_0" for i in range(n - 2)]
    pd.DataFrame({"segment_id": ids, "session_id": [300, 300, 301, 301], "dur_s": [1.0, 3.0, 2.0, 2.0],
                  "phq8_binary": [0, 0, 1, 1]}).to_parquet(root / "segments.parquet")
    rng = np.random.default_rng(0)
    pooled = rng.normal(size=(n, 13, 8)).astype(np.float32)
    for m in ("whisper_small", "hubert_base"):
        store.write_layered(root / m / "pooled.parquet", ids, {"mean": pooled, "std": np.abs(pooled)})
        (root / m / "frames").mkdir(parents=True)
        for s in (300, 301):
            save_file({f"{i}/layer_06": np.ones((5, 8), np.float16) for i in ids if i.startswith(str(s))},
                      root / m / "frames" / f"{s}.safetensors")
    store.write_layered(root / "text" / "roberta_base.parquet", ids, {"mean": pooled})
    store.write_vectors(root / "text" / "mpnet.parquet", ids, "emb", pooled[:, 0], dtype=np.float32)
    (root / "egemaps").mkdir()
    pd.DataFrame({"segment_id": ids, "F0": [1.0, 2, 3, 4]}).to_parquet(root / "egemaps" / "functionals.parquet")
    (root / "manifest.json").write_text(json.dumps({"config": {"models": {"frame_layers": [6]}}}))
    return ids, pooled


def test_roundtrip(tmp_path):
    ids, pooled = _make(tmp_path)
    ds = EmbeddingDataset(tmp_path)
    assert ds.segment_ids == ids
    np.testing.assert_allclose(ds.pooled("hubert_base", 3), pooled[:, 3], atol=1e-2)
    assert ds.pooled_all_layers("whisper_small", "std").shape == (4, 13, 8)
    assert ds.frames("hubert_base", ids[0], 6).shape == (5, 8)
    assert set(ds.session_frames("whisper_small", 301, 6)) == set(ids[2:])
    assert ds.text("roberta_base", 12).shape == (4, 8) and ds.text("mpnet").shape == (4, 8)
    assert list(ds.egemaps().index) == ids


def test_session_pool_weighted(tmp_path):
    _make(tmp_path)
    ds = EmbeddingDataset(tmp_path)
    X = np.arange(4, dtype=np.float32)[:, None]
    Xs, ys, sessions = ds.session_pool(X, weight_by_duration=True)
    np.testing.assert_allclose(Xs[:, 0], [0.75, 2.5])  # (0*1 + 1*3)/4, (2+3)/2
    assert list(ys) == [0, 1] and list(sessions) == [300, 301]


def test_audio_roundtrip(tmp_path):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf

    _make(tmp_path)
    rng = np.random.default_rng(0)
    full = (rng.normal(0, 0.1, 16000 * 3).clip(-1, 1) * 32767).astype(np.int16) / 32768.0
    (tmp_path / "audio" / "sessions").mkdir(parents=True)
    (tmp_path / "audio" / "segments").mkdir(parents=True)
    sf.write(tmp_path / "audio" / "sessions" / "300.flac", full, 16000, subtype="PCM_16")
    clips = []
    for a, b in [(0, 16000), (16000, 48000)]:
        buf = io.BytesIO()
        sf.write(buf, full[a:b], 16000, format="FLAC", subtype="PCM_16")
        clips.append({"bytes": buf.getvalue(), "path": "x.flac"})
    pq.write_table(pa.table({"segment_id": ["300_0000_0", "300_0001_0"], "audio": clips}),
                   tmp_path / "audio" / "segments" / "300.parquet")
    ds = EmbeddingDataset(tmp_path)
    wav, sr = ds.audio("300_0001_0")
    assert sr == 16000 and np.array_equal(wav, full[16000:48000].astype(np.float32))
    whole, _ = ds.session_audio(300)
    assert np.array_equal(whole, full.astype(np.float32))
