---
license: other
license_name: daic-woz-eula
pretty_name: DAIC-WOZ segment embeddings (sample)
tags: [speech, depression, embeddings, whisper, hubert, egemaps]
---

# DAIC-WOZ segment embeddings: sample

> **Restricted data.** These files are derived from the DAIC-WOZ corpus (USC ICT) and include
> participant transcripts and PHQ-8 labels. Access is limited to people who have signed the
> DAIC-WOZ EULA. Do not redistribute, make this repo public, or copy the files to other locations.

This is a test batch for the RFC *"DAIC-WOZ Embedding Generation"*. It covers {{N_SEGMENTS}} participant segments from
sessions {{SESSIONS}}: 2 per official split (train, dev, test), with 1 depressed (PHQ-8 ≥ 10) and 1 not depressed in each pair.
It exists to exercise the pipeline and probe code. It is far too small for meaningful results.

## Join key

Every table has `segment_id = "{session}_{turn_idx:04d}_{sub_idx}"`. All tables have the same rows in the same order.

## Files

| path | contents |
|---|---|
| `segments.parquet` | `segment_id, session_id, turn_idx, sub_idx, start_s, end_s, dur_s, text, merged_from, text_split_approx, split, phq8_score, phq8_binary, gender` |
| `whisper_small/pooled.parquet` | `mean_l00..mean_l12`, `std_l00..std_l12`: 768-d float16 per layer. Whisper-small **encoder** with padding frames trimmed |
| `hubert_base/pooled.parquet` | same layout, HuBERT Base (ls960), batch size 1 |
| `{whisper_small,hubert_base}/frames/<session>.safetensors` | frame-level float16 `[T, 768]` for layers 0, 6, 12, key `"<segment_id>/layer_XX"`, 50 frames/s |
| `egemaps/functionals.parquet` | 88 eGeMAPSv02 functionals (openSMILE) on the same segment audio |
| `text/roberta_base.parquet` | `mean_l00..mean_l12`: RoBERTa-base token-mean per hidden state over the segment transcript |
| `text/mpnet.parquet` | `emb`: all-mpnet-base-v2 sentence embedding (float32) |
| `manifest.json` | model ids and commit shas, code commit, full config, library versions, per-session segmentation stats |

Layer 0 is the pre-transformer output and layers 1–12 are the transformer blocks. For the Whisper encoder, layer 12 comes after the final layer norm.

## Segmentation

- Participant turns are taken from the transcript timestamps. Ellie's turns and any audio before her first line are dropped.
- Turns shorter than 1 s are merged into an adjacent participant turn when the gap is ≤ 0.5 s and no Ellie line falls between them. Otherwise they are dropped. Merges are listed in `merged_from` (transcript row indices).
- Segments longer than 30 s are split at the quietest 10 ms frame in the 3 s before the limit. Transcript words are divided between the pieces in proportion to time (`text_split_approx = true`).
- Audio is 16 kHz mono, with each model's own feature extractor and no extra amplitude normalization.
- Sessions with documented problems are excluded: 318, 321, 341, 342, 362, 373, 394, 398, 444, 451, 458, 460, 480.

## Loading

```python
from huggingface_hub import snapshot_download
import pandas as pd, numpy as np, pyarrow.parquet as pq

root = snapshot_download("GenerationGenerative/daic-woz-embeddings-sample", repo_type="dataset")
seg = pd.read_parquet(f"{root}/segments.parquet")
t = pq.read_table(f"{root}/hubert_base/pooled.parquet", columns=["segment_id", "mean_l09"])
X = np.stack(t.column("mean_l09").to_numpy(zero_copy_only=False)).astype(np.float32)  # [N, 768]
```

`gender` keeps the source coding from the DAIC-WOZ split CSVs (0/1).
