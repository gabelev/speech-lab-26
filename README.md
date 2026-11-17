# speech-lab-26

Code for work with the Speech Lab at Columbia: segment-level speech and text embeddings for the
**DAIC-WOZ** depression interview corpus. They support layer-wise probing and interpretability work
on Whisper, HuBERT, eGeMAPS and text representations.

The design follows the RFC *"DAIC-WOZ Embedding Generation"*. The current release is a **6-session sample**
for testing the pipeline and probe code. The full-corpus run comes next.

> **Data access.** DAIC-WOZ is released under a EULA from USC ICT. Raw audio, transcripts and every derived
> file (embeddings, transcripts, PHQ-8 labels) must stay with people who have signed it. This repo holds
> **code only**: `data/` and `outputs/` are git-ignored. The embeddings live in a **private** Hugging Face
> dataset repo.

---

## Reading the data

**Hugging Face repo (private):** https://huggingface.co/datasets/GenerationGenerative/daic-woz-embeddings-sample

Access goes through the `GenerationGenerative` organization: ask an org admin to add you, after you have signed the DAIC-WOZ EULA. If you use a fine-grained token, give it read access to `GenerationGenerative` repos.

### 1. Install and authenticate

```bash
git clone https://github.com/gabelev/speech-lab-26 && cd speech-lab-26
```

```bash
pip install -e .
```

```bash
huggingface-cli login   # or: export HF_TOKEN=hf_...  (a token with read access to the repo)
```

### 2. Load with `EmbeddingDataset`

`src/speechlab/load.py` wraps every table. On first use it downloads the repo into the
Hugging Face cache (about 1 GB, or about 45 MB with `frames=False`).

```python
from speechlab.load import EmbeddingDataset

ds = EmbeddingDataset.from_hub("GenerationGenerative/daic-woz-embeddings-sample")   # or EmbeddingDataset("outputs/sample/hf")

seg = ds.segments()                                    # one row per segment: times, text, split, PHQ-8, gender
X   = ds.pooled("hubert_base", layer=9)                # [N, 768] float32, mean over frames
S   = ds.pooled("whisper_small", layer=4, stat="std")  # [N, 768], std over frames
A   = ds.pooled_all_layers("whisper_small")            # [N, 13, 768]
F   = ds.frames("hubert_base", "303_0000_0", layer=6)  # [T, 768] at 50 frames/s (layers 0, 6, 12 only)
E   = ds.egemaps()                                     # DataFrame [N, 88], indexed by segment_id
T   = ds.text("roberta_base", layer=12)                # [N, 768]
M   = ds.text("mpnet")                                 # [N, 768]

# segment -> session (optionally weighted by segment duration)
Xs, y, sessions = ds.session_pool(X, weight_by_duration=True)
```

Every array's rows are in the same order as `ds.segments()`, so they line up by position.
`segment_id` is the shared key if you'd rather join DataFrames.

### 3. Example: a layer-wise probe

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

seg = ds.segments()
y, groups = seg.phq8_binary.values, seg.session_id.values
for layer in range(13):
    X = ds.pooled("hubert_base", layer)
    scores = np.zeros(len(y))
    for g in np.unique(groups):                     # leave-one-session-out
        tr, te = groups != g, groups == g
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    print(layer, round(roc_auc_score(y, scores), 3))
```

With 6 sessions, numbers like these test the code, not depression detection. A held-out session contains only one label, so the
AUC mostly reflects speaker and recording differences.

### Quick look and sanity checks

```bash
python examples/read_embeddings.py                       # summary of the HF repo
```

```bash
python scripts/validate.py --hf GenerationGenerative/daic-woz-embeddings-sample   # consistency checks + smoke probe
```

### Without this package

All files are plain parquet and safetensors:

```python
import pandas as pd, numpy as np, pyarrow.parquet as pq
from huggingface_hub import snapshot_download

root = snapshot_download("GenerationGenerative/daic-woz-embeddings-sample", repo_type="dataset")
seg = pd.read_parquet(f"{root}/segments.parquet")
col = pq.read_table(f"{root}/hubert_base/pooled.parquet", columns=["mean_l09"]).column(0).combine_chunks()
X = col.flatten().to_numpy().reshape(len(col), 768).astype(np.float32)
```

---

## Dataset layout

| file | contents |
|---|---|
| `segments.parquet` | `segment_id, session_id, turn_idx, sub_idx, start_s, end_s, dur_s, text, merged_from, text_split_approx, split, phq8_score, phq8_binary, gender` |
| `whisper_small/pooled.parquet` | `mean_l00 … mean_l12`, `std_l00 … std_l12`: one 768-d float16 vector per layer |
| `hubert_base/pooled.parquet` | same layout |
| `{model}/frames/<session>.safetensors` | float16 `[T, 768]` per segment for layers 0, 6, 12. Key `"<segment_id>/layer_XX"`, 50 frames/s |
| `egemaps/functionals.parquet` | `segment_id` + 88 eGeMAPSv02 functionals |
| `text/roberta_base.parquet` | `mean_l00 … mean_l12`: RoBERTa-base token-mean per hidden state |
| `text/mpnet.parquet` | `emb`: all-mpnet-base-v2 sentence embedding (float32) |
| `manifest.json` | model ids and commit shas, code commit, full config, library versions, per-session segmentation stats |

`segment_id = "{session}_{turn_idx:04d}_{sub_idx}"`. Layer 0 is the pre-transformer output and layers 1–12 are the
transformer blocks. For the Whisper encoder, layer 12 comes after the final layer norm.
`gender` keeps the 0/1 coding of the DAIC-WOZ split files.

### Current sample (6 sessions, 491 segments)

| split | session | PHQ-8 | depressed | gender | segments | speech (s) |
|---|---|---|---|---|---|---|
| train | 303 | 0 | 0 | 0 | 88 | 631 |
| train | 319 | 13 | 1 | 1 | 73 | 203 |
| dev | 302 | 4 | 0 | 1 | 66 | 190 |
| dev | 335 | 12 | 1 | 0 | 107 | 399 |
| test | 306 | 0 | 0 | 0 | 94 | 501 |
| test | 309 | 15 | 1 | 1 | 63 | 163 |

The dev pair's genders are flipped relative to train and test, so gender isn't confounded with the label across the sample.

---

## How the data is processed

Run end to end with `scripts/run_pipeline.py --config configs/sample.yaml [--push]`. All parameters are in the config.

### 1. Source data
- Each session comes from the DAIC-WOZ server as `<id>_P.zip`. Only `<id>_AUDIO.wav` (16 kHz mono) and `<id>_TRANSCRIPT.csv` are kept, and the zip is deleted.
- Labels come from `train_split_Depression_AVEC2017.csv`, `dev_split_Depression_AVEC2017.csv` and `full_test_split.csv`. The last is the labelled test split.
- **Excluded sessions:**
  - 342, 394, 398 and 460 are excluded by the corpus itself.
  - 373 and 444 have interruptions mid-interview.
  - 451, 458 and 480 have no Ellie transcript.
  - 318, 321, 341 and 362 have documented transcript timing offsets.

### 2. Segmentation (`src/speechlab/segment.py`)
Every feature type uses the same segments, so all tables join one to one.
1. **Participant turns only.** Ellie's lines are dropped, and so is anything before Ellie's first line (pre-interview setup audio).
2. **Short turns (< 1 s).**
   - A short turn is merged into the previous participant turn when the gap is ≤ 0.5 s and no Ellie line falls between them. If that fails, it's merged into the next turn under the same conditions. Otherwise it's dropped.
   - `merged_from` lists the source transcript rows of each segment.
   - In the sample, 185 short turns were dropped (about 112 s), mostly "um", "yeah", "no" and "mhm". *Depressed participants lose proportionally more of these, which is an open question for the full run.*
3. **Long turns (> 30 s)** are split at the quietest 10 ms frame (lowest RMS) in the 3 s before the 30 s mark, repeating until every piece is ≤ 30 s. Transcript words are divided between the pieces in proportion to time (`text_split_approx = true`).

### 3. Speech models (`src/speechlab/extract_ssl.py`)
- **Input:** 16 kHz mono, with each model's own feature extractor and no extra amplitude normalization.
- **Whisper-small (`openai/whisper-small`), encoder only.**
  - Whisper pads every input to 30 s, giving 1500 frames.
  - Only the first `ceil(n_samples / 320)` frames are kept, the ones covering real audio, before pooling.
- **HuBERT Base (`facebook/hubert-base-ls960`)** runs at batch size 1, so there's no padding.
- **Layers:** all 13 hidden states are kept for both models. Each layer is pooled to its **mean and population std** over frames and stored as float16.
- **Frame level:** frame-level hidden states are stored for layers **0, 6, 12** (float16), for lexical-erasure work.
- **Pinned versions:** model versions are pinned to the commit sha resolved at run time and recorded in `manifest.json`.

### 4. eGeMAPS (`src/speechlab/extract_other.py`)
openSMILE `eGeMAPSv02` functionals (88 features) on exactly the same segment audio.

### 5. Text (`src/speechlab/extract_other.py`)
- **RoBERTa-base:** all 13 hidden states, mean-pooled over non-special tokens. This mirrors the speech models' layer structure, so layer-wise probe curves can be compared.
- **all-mpnet-base-v2:** a sentence embedding, as a strong text baseline.

### 6. Publishing (`scripts/run_pipeline.py --push`)
Writes the dataset folder, `manifest.json` and a dataset card. It creates the HF dataset repo as **private** and refuses to upload unless the repo is private.

---

## Repository layout

```
configs/sample.yaml          sessions, segmentation rules, model ids, frame layers, HF repo
src/speechlab/
  daicwoz.py                 download, transcript and label loading
  segment.py                 participant-turn segmentation (shared by every extractor)
  extract_ssl.py             Whisper-small encoder and HuBERT Base hidden states + pooling
  extract_other.py           eGeMAPS, RoBERTa, mpnet
  store.py                   parquet writers (one column per layer)
  load.py                    EmbeddingDataset reader
scripts/run_pipeline.py        end-to-end pipeline (cached per session, resumable)
scripts/validate.py          consistency checks + smoke probe
scripts/dataset_card.md      README template for the HF repo
examples/read_embeddings.py  loading example
tests/                       segmentation, frame trimming, reader round-trip
```

## Reproducing the sample

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

```bash
python scripts/run_pipeline.py --config configs/sample.yaml           # about 11 min on 4 CPU cores
```

```bash
python scripts/validate.py outputs/sample/hf
```

```bash
python scripts/run_pipeline.py --config configs/sample.yaml --push    # needs an HF write token
```

**Run cost on 4 CPU cores:**
- Whisper-small takes about 1 s per segment, because it pads every input to 30 s.
- HuBERT takes about 0.25 s per segment.
- eGeMAPS and the text models take under a minute in total.

## Open decisions (before the full run)
- Small vs. extended DAIC-WOZ.
- The minimum segment duration and merge rule, given that short turns are dropped more often for depressed participants.
- Duration weighting when pooling segments to sessions (`session_pool(weight_by_duration=...)` supports both).
- Which layers to keep at frame level.
