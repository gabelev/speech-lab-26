# speech-lab-26

Code for work with the Speech Lab at Columbia: segment-level speech and text embeddings for the
**DAIC-WOZ** depression interview corpus. They support layer-wise probing and interpretability work
on Whisper, HuBERT, eGeMAPS and text representations.

The design follows the RFC *"DAIC-WOZ Embedding Generation"*. There are two releases:
- **Full corpus:** 179 sessions and 22,083 segments, in `GenerationGenerative/daic-woz-embeddings`.
- **Sample:** 6 sessions, for quick tests of the pipeline and probe code, in `GenerationGenerative/daic-woz-embeddings-sample`.

> **Data access.** DAIC-WOZ is released under a EULA from USC ICT. Raw audio, transcripts and every derived
> file (embeddings, transcripts, PHQ-8 labels) must stay with people who have signed it. This repo holds
> **code only**: `data/` and `outputs/` are git-ignored. The embeddings live in a **private** Hugging Face
> dataset repo.

---

## Reading the data

**Hugging Face repos (private):**
- **Full corpus:** https://huggingface.co/datasets/GenerationGenerative/daic-woz-embeddings
  - 179 sessions, 22,083 segments, 21.9 h of participant speech.
  - About 2 GB without frames, 38 GB with them.
- **Sample:** https://huggingface.co/datasets/GenerationGenerative/daic-woz-embeddings-sample
  - 6 sessions, about 1 GB.

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

`src/speechlab/load.py` wraps every table. On first use it downloads the repo into the Hugging Face cache.
- **Full corpus:** use `frames=False` unless you need frames. The pooled tables take about 2 GB, and the frames add 36 GB.
- **Sample:** about 1 GB, or about 45 MB with `frames=False`.

```python
from speechlab.load import EmbeddingDataset

ds = EmbeddingDataset.from_hub("GenerationGenerative/daic-woz-embeddings", frames=False)
# sample: EmbeddingDataset.from_hub("GenerationGenerative/daic-woz-embeddings-sample"); local: EmbeddingDataset("outputs/full/hf")

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
from sklearn.model_selection import GroupKFold

seg = ds.segments()
y, groups = seg.phq8_binary.values, seg.session_id.values
for layer in range(13):
    X = ds.pooled("hubert_base", layer)
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):  # never split a session across train and test
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    print(layer, round(roc_auc_score(y, scores), 3))
```

Segment-level AUCs pool many segments per speaker. For depression results, report session-level scores on the official
train/dev/test splits (`ds.session_pool`). On the 6-session sample, numbers like these only show that the code runs.

### Quick look and sanity checks

```bash
python examples/read_embeddings.py                       # summary of the HF repo
```

```bash
python scripts/validate.py --hf GenerationGenerative/daic-woz-embeddings   # consistency checks + probe (frames for 3 sessions)
```

### Without this package

All files are plain parquet and safetensors:

```python
import pandas as pd, numpy as np, pyarrow.parquet as pq
from huggingface_hub import snapshot_download

root = snapshot_download("GenerationGenerative/daic-woz-embeddings", repo_type="dataset", ignore_patterns=["*/frames/*"])
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

### Full corpus (179 sessions, 22,083 segments)

| split | sessions | depressed | segments | speech (h) |
|---|---|---|---|---|
| train | 102 | 28 | 11,564 | 11.7 |
| dev | 32 | 11 | 4,325 | 4.1 |
| test | 45 | 14 | 6,194 | 6.0 |

- **Segment length:** median 2.8 s, mean 3.6 s, range 1–29.4 s.
- **Merged and dropped turns:** of 30,532 participant transcript rows, 984 were merged into a neighbouring turn and 7,341 short turns (24%) were dropped. See the open decisions below.
- **Session 440 (dev) is missing:** the DAIC-WOZ server's copy of its zip is truncated, with no zip directory and no transcript. It is listed under `config.failed_sessions` in `manifest.json`.

**First-look baseline** (not a result to report):
- **Setup:** logistic regression trained on session means of the 102 train sessions, weighted by segment duration, scored on dev and test. AUC is for PHQ-8 ≥ 10, over C ∈ {0.001, 0.01, 0.1}.
- **Text:** RoBERTa layers 6 and 12 reach 0.69–0.79 on test, and mpnet about 0.83. This confirms the labels and segment joins line up.
- **Speech:** Whisper, HuBERT and eGeMAPS session means stay near chance on test (about 0.4–0.6), though Whisper's last layer reaches 0.61–0.81 on dev.
- **Segment level:** `scripts/validate.py`'s segment-level probe is about 0.5 for every model.
- **Takeaway:** mean-pooled speech embeddings plus a linear probe aren't enough. This is the gap the layer-wise and frame-level work is meant to explore.

### Sample (6 sessions, 491 segments)

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

Run end to end with `scripts/run_pipeline.py --config configs/{sample,full}.yaml [--push]`. All parameters are in the config.

### 1. Source data
- Each session comes from the DAIC-WOZ server as `<id>_P.zip`. Only `<id>_AUDIO.wav` (16 kHz mono) and `<id>_TRANSCRIPT.csv` are kept, and the zip is deleted.
- Labels come from `train_split_Depression_AVEC2017.csv`, `dev_split_Depression_AVEC2017.csv` and `full_test_split.csv`. The last is the labelled test split.
- **Excluded sessions:**
  - 342, 394, 398 and 460 are excluded by the corpus itself.
  - 373 and 444 have interruptions mid-interview.
  - 451, 458 and 480 have no Ellie transcript.
  - 318, 321, 341 and 362 have documented transcript timing offsets.
  - 440's zip on the server is truncated. The pipeline skips any session whose zip can't be read and records it in the manifest.

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
configs/sample.yaml          6-session sample: sessions, segmentation rules, model ids, frame layers, HF repo
configs/full.yaml            full corpus: all sessions, pinned model revisions, incremental frame upload
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

## Reproducing

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

```bash
python scripts/run_pipeline.py --config configs/full.yaml --push      # full corpus, resumable
```

**Full run:** the full corpus took about 9 h of processing on 4 CPU cores.
- Each session is processed separately and marked done in the cache, so a rerun resumes after the last finished session.
- Frame files are uploaded every 10 sessions and then deleted locally.
- On one GPU it would take about 1.5 h, mostly downloading.
- `--shard i/n` splits the sessions across machines; once all shards are done, run `--assemble-only`.

**Run cost on 4 CPU cores:**
- Whisper-small takes about 1 s per segment, because it pads every input to 30 s.
- HuBERT takes about 0.25 s per segment.
- eGeMAPS and the text models take under a minute in total.

## Open decisions
- Small vs. extended DAIC-WOZ.
- The minimum segment duration and merge rule, given that short turns are dropped more often for depressed participants.
- Duration weighting when pooling segments to sessions (`session_pool(weight_by_duration=...)` supports both).
- Which layers to keep at frame level.
