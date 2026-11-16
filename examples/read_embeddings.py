"""Example: load the sample embeddings and print what's there.

    python examples/read_embeddings.py                      # from the private HF repo
    python examples/read_embeddings.py outputs/sample/hf    # from a local output folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab.load import EmbeddingDataset  # noqa: E402

ds = EmbeddingDataset(sys.argv[1]) if len(sys.argv) > 1 else EmbeddingDataset.from_hub()

seg = ds.segments()
print(f"{len(seg)} segments from {seg.session_id.nunique()} sessions")
print(seg.groupby(["session_id", "split", "phq8_binary", "gender"]).dur_s.agg(["count", "sum"]).round(1), "\n")

print("models:", {k: v["id"] for k, v in ds.manifest["models"].items()})
print("code commit:", ds.manifest["code_commit"], "\n")

for model in ("whisper_small", "hubert_base"):
    A = ds.pooled_all_layers(model)
    print(f"{model}: pooled {A.shape}, layer norms", np.linalg.norm(A, axis=2).mean(0).round(1))

sid = seg.segment_id.iloc[0]
for layer in ds.frame_layers():
    print(f"frames hubert_base {sid} layer {layer}: {ds.frames('hubert_base', sid, layer).shape}")

print("egemaps:", ds.egemaps().shape, "| roberta L12:", ds.text("roberta_base", 12).shape,
      "| mpnet:", ds.text("mpnet").shape)

Xs, ys, sessions = ds.session_pool(ds.pooled("hubert_base", 9), weight_by_duration=True)
print(f"session-level HuBERT L9: {Xs.shape}, labels {dict(zip(sessions.tolist(), ys.tolist()))}")
