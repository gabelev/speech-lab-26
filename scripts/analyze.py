"""Descriptive statistics for the raw DAIC-WOZ data and the embedding dataset.

    python scripts/analyze.py --hf GenerationGenerative/daic-woz-embeddings --revision v2 \
        --root outputs/full_v2/hf --v1 v1.0 --out reports/v2

Writes ``<out>/STATS.md``, ``<out>/stats.json`` and ``<out>/figures/*.png``.

Everything written is aggregate: no transcript text and no per-session label rows, so the
report can sit in the code repo. Tables are read from ``--root`` when given (a local
output folder), otherwise from the hub. Audio and frame files are fetched from the hub one
session at a time when they are not local (the pipeline deletes them after uploading).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import mean_squared_error, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechlab import daicwoz  # noqa: E402
from speechlab.load import EmbeddingDataset  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-3, text and surface tokens, blue ramp
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MODEL_COLORS = {"whisper_small": BLUE, "hubert_base": ORANGE, "roberta_base": AQUA}
MODEL_NAMES = {"whisper_small": "Whisper-small enc.", "hubert_base": "HuBERT Base", "roberta_base": "RoBERTa-base"}
GROUP_COLORS = {0: BLUE, 1: ORANGE}
GROUP_NAMES = {0: "PHQ-8 < 10", 1: "PHQ-8 ≥ 10"}

FILLERS = {"um", "uh", "umm", "uhm", "uhh", "mm", "mmm", "mhm", "hmm", "hm", "mm-hmm", "uh-huh", "er", "ah"}
HEDGES = ["i don't know", "i guess", "kind of", "sort of", "maybe", "i think", "you know", "i mean", "probably"]
NONVERBAL = ["laughter", "sigh", "clears throat", "deep breath", "sniffle", "sniff", "snort", "tisk", "cough"]
TAG_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")
WORD_RE = re.compile(r"[a-z]+(?:[-'][a-z]+)*")


# --------------------------------------------------------------------------- plumbing
def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    })


class Files:
    """Local folder first, then the hub."""

    def __init__(self, root: Path | None, repo: str | None, revision: str | None):
        self.root, self.repo, self.revision = root, repo, revision

    def get(self, rel: str) -> Path | None:
        if self.root is not None and (self.root / rel).exists():
            return self.root / rel
        if self.repo is None:
            return None
        from huggingface_hub import hf_hub_download

        try:
            return Path(hf_hub_download(self.repo, rel, repo_type="dataset", revision=self.revision))
        except Exception:
            return None

    def tables(self) -> Path:
        if self.root is not None and (self.root / "segments.parquet").exists():
            return self.root
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(self.repo, repo_type="dataset", revision=self.revision,
                                      ignore_patterns=["*/frames/*", "audio/*"]))


def mw(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney p and rank-biserial r (positive = higher in b)."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    u, p = mannwhitneyu(b, a, alternative="two-sided")
    return float(p), float(2 * u / (len(a) * len(b)) - 1)


def bh(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q.tolist()


def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:,.{nd}f}"


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pctl(x, qs=(5, 25, 50, 75, 95)) -> dict:
    x = np.asarray(x, float)
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"mean": float(x.mean()), "n": int(len(x))}


# --------------------------------------------------------------------------- raw transcripts
def transcript_stats(data_dir: Path, sessions: list[int]) -> tuple[pd.DataFrame, np.ndarray]:
    rows, turn_durs = [], []
    for s in sessions:
        tr = daicwoz.load_transcript(data_dir, s)
        sp = tr.speaker.str.lower()
        first = tr.loc[sp == "ellie", "start_time"].min()
        tr = tr[tr.start_time >= first].reset_index(drop=True)
        sp = tr.speaker.str.lower()
        dur = (tr.stop_time - tr.start_time).clip(lower=0)
        P, E = tr[sp == "participant"], tr[sp == "ellie"]
        pd_ = dur[sp == "participant"]
        turn_durs += pd_[pd_ > 0].tolist()
        text = " ".join(P.value.str.lower())
        clean = TAG_RE.sub(" ", text)
        words = [w for w in WORD_RE.findall(clean) if w not in ("xxx", "scrubbed_entry")]
        n_words = max(len(words), 1)
        lat = [tr.start_time[i + 1] - tr.stop_time[i] for i in range(len(tr) - 1)
               if sp[i] == "ellie" and sp[i + 1] == "participant"]
        talk = float(pd_.sum())
        rows.append({
            "session_id": s,
            "interview_min": float((tr.stop_time.max() - first) / 60),
            "participant_talk_min": talk / 60,
            "participant_share": talk / max(talk + float(dur[sp == "ellie"].sum()), 1e-9),
            "participant_turns": len(P),
            "ellie_turns": len(E),
            "words": len(words),
            "words_per_turn": len(words) / max(len(P), 1),
            "speech_rate_wps": len(words) / max(talk, 1e-9),
            "median_turn_s": float(pd_[pd_ > 0].median()),
            "short_turn_share": float((pd_[pd_ > 0] < 1.0).mean()),
            "response_latency_s": float(np.median(lat)) if lat else np.nan,
            "fillers_per_100w": 100 * sum(w in FILLERS for w in words) / n_words,
            "hedges_per_100w": 100 * sum(clean.count(h) for h in HEDGES) / n_words,
            "i_me_my_per_100w": 100 * sum(w in ("i", "me", "my", "myself", "i'm") for w in words) / n_words,
            "nonverbal_per_min": sum(text.count(f"<{t}>") + text.count(f"[{t}]") for t in NONVERBAL) / max(talk / 60, 1e-9),
            "laughter_per_min": (text.count("<laughter>") + text.count("[laughter]")) / max(talk / 60, 1e-9),
            "unintelligible_per_100w": 100 * text.count("xxx") / n_words,
        })
    return pd.DataFrame(rows).set_index("session_id"), np.asarray(turn_durs)


# --------------------------------------------------------------------------- audio
def audio_stats(files: Files, seg: pd.DataFrame, sessions: list[int]) -> pd.DataFrame:
    rows = []
    for k, s in enumerate(sessions):
        p = files.get(f"audio/sessions/{s}.flac")
        if p is None:
            continue
        x, sr = sf.read(p, dtype="float32")
        hop = int(0.025 * sr)
        n = len(x) // hop
        fr = x[: n * hop].reshape(n, hop)
        e_db = 10 * np.log10((fr.astype(np.float64) ** 2).mean(1) + 1e-12)
        mask = np.zeros(n, bool)
        for iv in seg.loc[seg.session_id == s, "intervals"]:
            for a, b in iv:
                mask[int(a * sr) // hop: int(b * sr) // hop] = True
        speech_db = 10 * np.log10(np.mean(10 ** (e_db[mask] / 10))) if mask.any() else np.nan
        floor_db = float(np.percentile(e_db, 10))
        rows.append({
            "session_id": s, "duration_min": len(x) / sr / 60,
            "rms_dbfs": float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)),
            "participant_speech_dbfs": float(speech_db), "noise_floor_dbfs": floor_db,
            "snr_db": float(speech_db - floor_db), "peak_dbfs": float(20 * np.log10(np.abs(x).max() + 1e-12)),
            "clipped_pct": float(100 * (np.abs(x) >= 0.999).mean()), "dc_offset": float(x.mean()),
        })
        if files.root is None or not (files.root / f"audio/sessions/{s}.flac").exists():
            try:
                p.unlink()  # hub cache copy; keep disk flat
            except OSError:
                pass
        if (k + 1) % 20 == 0:
            print(f"  audio {k + 1}/{len(sessions)}", flush=True)
    return pd.DataFrame(rows).set_index("session_id") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- embeddings
def layer_geometry(X: np.ndarray, sess: np.ndarray, standardize: bool = False) -> dict:
    X = X.astype(np.float64)
    if standardize:
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    norms = np.linalg.norm(X, axis=1)
    U = X / (norms[:, None] + 1e-12)
    s = U.sum(0)
    n = len(X)
    aniso = float((s @ s - n) / (n * (n - 1)))
    Xc = X - X.mean(0)
    ev = np.clip(np.linalg.eigvalsh(Xc.T @ Xc / n)[::-1], 0, None)
    cum = np.cumsum(ev) / ev.sum()
    eff_rank = float(np.exp(-(p := ev / ev.sum())[p > 0] @ np.log(p[p > 0])))
    total = float((Xc ** 2).sum(1).mean())
    df = pd.DataFrame(Xc)
    means = df.groupby(sess).mean().values
    counts = pd.Series(sess).value_counts().reindex(pd.unique(sess)).values
    order = pd.unique(sess)
    means = df.groupby(sess).mean().loc[order].values
    between = float((counts[:, None] * means ** 2).sum() / n)
    return {"norm": float(norms.mean()), "anisotropy": aniso, "pca90": int(np.searchsorted(cum, 0.9) + 1),
            "effective_rank": eff_rank, "between_session_share": between / total}


def duration_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    clip = FunctionTransformer(lambda z: np.clip(z, -5, 5))  # eGeMAPS has extreme outliers
    pred = cross_val_predict(make_pipeline(StandardScaler(), clip, Ridge(alpha=100.0)), X, y,
                             groups=groups, cv=GroupKFold(5))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def session_probe(Xs: np.ndarray, lab: pd.DataFrame, rng: np.random.Generator) -> dict:
    """PHQ-8 probes on session vectors with the official splits. Hyperparameters are picked
    by 5-fold CV inside the training sessions only."""
    out = {}
    for name, tr_splits, te_split in [("dev", ["train"], "dev"), ("test", ["train", "dev"], "test")]:
        tr, te = lab.split.isin(tr_splits).values, (lab.split == te_split).values
        yb, ys = lab.phq8_binary.values, lab.phq8_score.values.astype(float)
        clf = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                           {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1]}, scoring="roc_auc",
                           cv=StratifiedKFold(5, shuffle=True, random_state=0)).fit(Xs[tr], yb[tr])
        prob = clf.predict_proba(Xs[te])[:, 1]
        reg = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [1, 10, 100, 1e3, 1e4]},
                           scoring="neg_root_mean_squared_error", cv=5).fit(Xs[tr], ys[tr])
        rmse = float(np.sqrt(mean_squared_error(ys[te], reg.predict(Xs[te]))))
        auc = float(roc_auc_score(yb[te], prob))
        boot = []
        idx = np.arange(te.sum())
        for _ in range(500):
            b = rng.choice(idx, len(idx))
            if len(np.unique(yb[te][b])) == 2:
                boot.append(roc_auc_score(yb[te][b], prob[b]))
        out[name] = {"auc": auc, "auc_lo": float(np.percentile(boot, 2.5)), "auc_hi": float(np.percentile(boot, 97.5)),
                     "rmse": rmse, "rmse_baseline": float(np.sqrt(np.mean((ys[te] - ys[tr].mean()) ** 2)))}
    return out


def gender_auc(Xs: np.ndarray, g: np.ndarray) -> float:
    if len(np.unique(g)) < 2 or min(np.bincount(g)) < 5:
        return float("nan")
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)), Xs, g,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return float(roc_auc_score(g, p))


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", help="HF dataset repo (for tables if --root is missing, and for audio/frames)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--root", type=Path, default=None, help="local output folder (outputs/full_v2/hf)")
    ap.add_argument("--v1", default=None, help="revision of the previous release to compare segments with")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/daicwoz")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/v2")
    ap.add_argument("--frame-sessions", type=int, default=5)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    style()
    figs = args.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    files = Files(args.root, args.hf, args.revision)
    ds = EmbeddingDataset(files.tables())
    seg = ds.segments()
    man = ds.manifest
    sessions = [int(s) for s in man["sessions"]]
    lab = seg.groupby("session_id")[["split", "phq8_score", "phq8_binary", "gender"]].first().loc[sessions]
    R: dict = {"dataset_version": man.get("dataset_version"), "n_sessions": len(sessions), "n_segments": len(seg)}
    T: dict[str, list[dict]] = {}  # tidy tables for the report artifact (data/*.json)
    md: list[str] = []
    title = f"DAIC-WOZ embeddings v{man.get('dataset_version') or '?'}: descriptive statistics"
    md += [f"# {title}", "",
           f"Generated by `scripts/analyze.py` from dataset version {man.get('dataset_version')} "
           f"(code {man['code_commit'][:7]}). The figures are in `figures/`, and every number is also in `stats.json`.", "",
           "> All figures here are aggregates. There's no transcript text and no per-session label table, so this file "
           "can live in the code repo. Comparisons between depressed (PHQ-8 ≥ 10) and non-depressed participants "
           "are descriptive associations on a small sample, not findings. p-values are uncorrected Mann–Whitney tests, "
           "q-values are Benjamini–Hochberg across the metrics in each table, and r is the rank-biserial effect size "
           "(positive means higher in the depressed group).", ""]

    # ---- corpus --------------------------------------------------------------------
    md += ["## 1. Corpus", ""]
    ct = lab.groupby("split").agg(sessions=("phq8_binary", "size"), depressed=("phq8_binary", "sum"),
                                  gender_1=("gender", "sum"), phq8_mean=("phq8_score", "mean"),
                                  phq8_median=("phq8_score", "median"))
    ct = ct.loc[[s for s in ["train", "dev", "test"] if s in ct.index]]
    R["corpus"] = ct.reset_index().to_dict("records")
    T["corpus"] = [{"split": i, "sessions": int(r.sessions), "depressed": int(r.depressed), "gender_1": int(r.gender_1),
                    "phq8_mean": round(float(r.phq8_mean), 2), "phq8_median": float(r.phq8_median)}
                   for i, r in ct.iterrows()]
    T["phq8_hist"] = [{"score": int(k), "sessions": int((lab.phq8_score == k).sum()),
                       "depressed": int(k >= 10)} for k in range(25)]
    md += [md_table(["split", "sessions", "PHQ-8 ≥ 10", "gender = 1", "PHQ-8 mean", "PHQ-8 median"],
                    [[i, int(r.sessions), int(r.depressed), int(r.gender_1), fmt(r.phq8_mean, 1),
                      fmt(r.phq8_median, 0)]
                     for i, r in ct.iterrows()]), ""]
    xt = pd.crosstab(lab.gender, lab.phq8_binary)
    R["gender_by_label"] = {f"gender_{g}": {f"phq8_binary_{b}": int(xt.loc[g, b]) for b in xt.columns} for g in xt.index}
    md += [f"Gender (source coding 0/1) × PHQ-8 ≥ 10: " + "; ".join(
        f"gender {g}: {int(xt.loc[g].get(1, 0))} of {int(xt.loc[g].sum())} depressed" for g in xt.index) + ".", ""]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bins = np.arange(-0.5, 25.5, 1)
    ax.hist(lab.phq8_score, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(9.5, color=INK2, lw=1, ls="--")
    ax.text(9.8, ax.get_ylim()[1] * 0.92, "PHQ-8 ≥ 10", color=INK2, fontsize=8)
    ax.set(xlabel="PHQ-8 score", ylabel="sessions", title="PHQ-8 scores")
    fig.tight_layout()
    fig.savefig(figs / "phq8_hist.png", dpi=150)
    plt.close(fig)
    md += ["![PHQ-8 distribution](figures/phq8_hist.png)", ""]

    # ---- raw transcripts -------------------------------------------------------------
    md += ["## 2. Raw transcripts (before segmentation)", "",
           "These stats are computed from the original transcript rows, from Ellie's first line onward. "
           "Talk time is the sum of transcript row durations.", ""]
    ts, turn_durs = transcript_stats(args.data_dir, sessions)
    R["transcript_sessions"] = {c: pctl(ts[c].dropna()) for c in ts.columns}
    R["transcript_turn_durations_s"] = pctl(turn_durs) | {"share_lt_1s": float((turn_durs < 1).mean())}
    labels_ = {
        "interview_min": "interview length (min)", "participant_talk_min": "participant talk (min)",
        "participant_share": "participant share of talk", "participant_turns": "participant turns",
        "ellie_turns": "Ellie turns", "words": "participant words", "words_per_turn": "words per turn",
        "speech_rate_wps": "speech rate (words/s of talk)", "median_turn_s": "median turn length (s)",
        "short_turn_share": "share of turns < 1 s", "response_latency_s": "response latency after Ellie (s, median)",
        "fillers_per_100w": "fillers per 100 words (um, uh, mhm, ...)",
        "hedges_per_100w": "hedges per 100 words (I don't know, kind of, maybe, ...)",
        "i_me_my_per_100w": "first-person singular per 100 words",
        "nonverbal_per_min": "non-verbal tags per min of talk", "laughter_per_min": "laughter per min of talk",
        "unintelligible_per_100w": "unintelligible (xxx) per 100 words",
    }

    def group_table(df: pd.DataFrame, names: dict, key: str) -> None:
        recs = []
        for c, nm in names.items():
            if c not in df:
                continue
            v = df[c].astype(float)
            a = v[lab.loc[v.index, "phq8_binary"] == 0].values
            b = v[lab.loc[v.index, "phq8_binary"] == 1].values
            p, r = mw(a, b)
            recs.append({"metric_id": c, "metric": nm, "median": float(np.nanmedian(v)),
                         "p5": float(np.nanpercentile(v, 5)), "p95": float(np.nanpercentile(v, 95)),
                         "median_not_depressed": float(np.nanmedian(a)) if len(a) else None,
                         "median_depressed": float(np.nanmedian(b)) if len(b) else None, "r": r, "p": p})
        for rec, qq in zip(recs, bh([x["p"] for x in recs])):
            rec["q"] = qq
        T[key] = recs
        R[key] = {x["metric_id"]: x for x in recs}
        out = [[x["metric"], fmt(x["median"]), f"{fmt(x['p5'])} – {fmt(x['p95'])}", fmt(x["median_not_depressed"]),
                fmt(x["median_depressed"]), fmt(x["r"]), fmt(x["p"], 3), fmt(x["q"], 3)] for x in recs]
        md.extend([md_table(["metric (per session)", "median", "5th–95th pct", "median PHQ<10", "median PHQ≥10",
                             "r", "p", "q"], out), ""])

    group_table(ts, labels_, "transcript_by_group")
    md += [f"**Turn lengths** (all {len(turn_durs):,} participant rows): median {fmt(np.median(turn_durs))} s, "
           f"{fmt(100 * (turn_durs < 1).mean(), 1)}% shorter than 1 s, 95th percentile {fmt(np.percentile(turn_durs, 95), 1)} s.", ""]

    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, c in zip(axes.flat, ["interview_min", "participant_talk_min", "speech_rate_wps",
                                 "response_latency_s", "fillers_per_100w", "short_turn_share"]):
        for gval in (0, 1):
            v = ts.loc[lab.index[lab.phq8_binary == gval], c].dropna()
            jitter = rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(np.full(len(v), gval) + jitter, v, s=10, color=GROUP_COLORS[gval], alpha=0.6, linewidths=0,
                       zorder=3)
            if len(v):
                ax.hlines(np.median(v), gval - 0.3, gval + 0.3, color=INK, lw=1.5, zorder=2)
        ax.set_xticks([0, 1], [GROUP_NAMES[0], GROUP_NAMES[1]])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(labels_[c], fontsize=8.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Per-session transcript metrics by group (bar = median)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs / "transcript_by_group.png", dpi=150)
    plt.close(fig)
    md += ["![transcript metrics by group](figures/transcript_by_group.png)", ""]

    # ---- audio -------------------------------------------------------------------------
    if not args.no_audio:
        print("audio stats", flush=True)
        au = audio_stats(files, seg, sessions)
        if len(au):
            md += ["## 3. Raw audio (full session recordings)", "",
                   "Levels are in dBFS from 25 ms frames. **Participant speech** is the energy inside segment intervals. "
                   "The **noise floor** is the 10th percentile of frame energy over the whole recording, and **SNR** is "
                   "participant speech minus noise floor, which is a rough figure.", ""]
            R["audio_sessions"] = {c: pctl(au[c]) for c in au.columns}
            group_table(au, {"duration_min": "recording length (min)", "rms_dbfs": "overall level (dBFS)",
                             "participant_speech_dbfs": "participant speech level (dBFS)",
                             "noise_floor_dbfs": "noise floor (dBFS)", "snr_db": "SNR estimate (dB)",
                             "peak_dbfs": "peak (dBFS)", "clipped_pct": "clipped samples (%)",
                             "dc_offset": "DC offset"}, "audio_by_group")
            n_clip = int((au.clipped_pct > 0.01).sum())
            md += [f"{n_clip} of {len(au)} sessions have more than 0.01% clipped samples.", ""]
            fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
            axes[0].hist(au.snr_db, bins=25, color=BLUE, edgecolor=SURFACE)
            axes[0].set(xlabel="dB", ylabel="sessions", title="SNR estimate")
            axes[1].hist(au.participant_speech_dbfs, bins=25, color=BLUE, edgecolor=SURFACE)
            axes[1].set(xlabel="dBFS", title="Participant speech level")
            fig.tight_layout()
            fig.savefig(figs / "audio_quality.png", dpi=150)
            plt.close(fig)
            md += ["![audio quality](figures/audio_quality.png)", ""]

    # ---- segments ----------------------------------------------------------------------
    md += ["## 4. Segments", ""]
    st = man["segmentation_stats"]
    agg = {k: int(sum(v[k] if isinstance(v[k], int) else len(v[k]) for v in st.values()))
           for k in ["participant_rows", "rows_pre_interview", "rows_nonpositive", "rows_short", "rows_merged",
                     "turns_kept", "turns_spliced", "turns_split", "segments"] if k in next(iter(st.values()))}
    R["segmentation_totals"] = agg
    per_sess = seg.groupby("session_id").size()
    words = seg.text.str.split().str.len()
    R["segments"] = {"dur_s": pctl(seg.dur_s), "per_session": pctl(per_sess), "words": pctl(words),
                     "spliced": int(seg.spliced.sum()) if "spliced" in seg else None,
                     "with_short_turns": int((seg.n_short_merged > 0).sum()) if "n_short_merged" in seg else None,
                     "shorter_than_1s": int((seg.dur_s < 1).sum())}
    rows = [["segments", fmt(len(seg))], ["segments per session (median, range)",
                                          f"{int(per_sess.median())} ({per_sess.min()}–{per_sess.max()})"],
            ["segment length (s): median / mean / 95th pct",
             f"{fmt(seg.dur_s.median())} / {fmt(seg.dur_s.mean())} / {fmt(seg.dur_s.quantile(.95))}"],
            ["words per segment (median)", fmt(float(words.median()), 0)],
            ["participant speech in segments (h)", fmt(seg.dur_s.sum() / 3600, 1)]]
    for k, nm in [("participant_rows", "participant transcript rows"), ("rows_pre_interview", "rows before the interview (dropped)"),
                  ("rows_nonpositive", "empty rows (dropped)"), ("rows_short", "short rows (< 1 s), all kept"),
                  ("rows_merged", "rows merged into another turn"), ("turns_spliced", "turns spliced across an Ellie line"),
                  ("turns_split", "turns split at 30 s")]:
        if k in agg:
            rows.append([nm, fmt(agg[k])])
    if "spliced" in seg:
        rows.append(["segments containing a short turn", fmt(int((seg.n_short_merged > 0).sum()))])
        rows.append(["segments shorter than 1 s", fmt(int((seg.dur_s < 1).sum()))])
    v1seg = None
    if args.v1 and args.hf:
        from huggingface_hub import hf_hub_download

        v1seg = pd.read_parquet(hf_hub_download(args.hf, "segments.parquet", repo_type="dataset", revision=args.v1))
        v1seg = v1seg[v1seg.session_id.isin(sessions)]
        R["segments_v1"] = {"n": len(v1seg), "dur_s": pctl(v1seg.dur_s), "speech_h": float(v1seg.dur_s.sum() / 3600)}
        md += [f"**Compared with {args.v1}** (the same sessions):", "",
               md_table(["", f"v{man.get('dataset_version')}", args.v1],
                        [["segments", fmt(len(seg)), fmt(len(v1seg))],
                         ["segment length median (s)", fmt(seg.dur_s.median()), fmt(v1seg.dur_s.median())],
                         ["segments shorter than 2 s", fmt(int((seg.dur_s < 2).sum())), fmt(int((v1seg.dur_s < 2).sum()))],
                         ["participant speech in segments (h)", fmt(seg.dur_s.sum() / 3600, 2),
                          fmt(v1seg.dur_s.sum() / 3600, 2)]]), ""]
    md += [md_table(["", "value"], rows), ""]
    bins = np.logspace(np.log10(0.2), np.log10(30), 50)
    h2 = np.histogram(seg.dur_s.clip(0.2, 30), bins)[0]
    hr = np.histogram(np.clip(turn_durs, 0.2, 30), bins)[0]
    h1 = np.histogram(v1seg.dur_s.clip(0.2, 30), bins)[0] if v1seg is not None else np.zeros_like(h2)
    T["length_hist"] = [{"bin": i, "from_s": round(float(bins[i]), 3), "to_s": round(float(bins[i + 1]), 3),
                         "v2_segments": int(h2[i]), "v1_segments": int(h1[i]), "transcript_rows": int(hr[i])}
                        for i in range(len(bins) - 1)]
    cmp_rows = [("segments", len(seg), len(v1seg) if v1seg is not None else None),
                ("median_s", float(seg.dur_s.median()), float(v1seg.dur_s.median()) if v1seg is not None else None),
                ("under_2s", int((seg.dur_s < 2).sum()), int((v1seg.dur_s < 2).sum()) if v1seg is not None else None),
                ("speech_h", float(seg.dur_s.sum() / 3600), float(v1seg.dur_s.sum() / 3600) if v1seg is not None else None),
                ("per_session_median", float(per_sess.median()),
                 float(v1seg.groupby("session_id").size().median()) if v1seg is not None else None)]
    T["segments_compare"] = [{"measure": a, "v2": b, "v1": c} for a, b, c in cmp_rows]
    T["segmentation_totals"] = [{"measure": k, "count": v} for k, v in agg.items()] + [
        {"measure": "segments_spliced", "count": int(seg.spliced.sum()) if "spliced" in seg else 0},
        {"measure": "segments_with_short_turns", "count": int((seg.n_short_merged > 0).sum()) if "n_short_merged" in seg else 0},
        {"measure": "segments_under_1s", "count": int((seg.dur_s < 1).sum())},
        {"measure": "transcript_rows_under_1s", "count": int((turn_durs < 1).sum())},
        {"measure": "transcript_rows_total", "count": int(len(turn_durs))}]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.hist(seg.dur_s, bins=bins, histtype="step", color=BLUE, lw=2, label=f"v{man.get('dataset_version')}")
    if v1seg is not None:
        ax.hist(v1seg.dur_s, bins=bins, histtype="step", color=ORANGE, lw=2, label=args.v1)
    ax.hist(turn_durs, bins=bins, histtype="step", color=INK2, lw=1, ls="--", label="raw transcript rows")
    ax.set_xscale("log")
    ax.set(xlabel="seconds (log scale)", ylabel="count", title="Segment and turn lengths")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "segment_lengths.png", dpi=150)
    plt.close(fig)
    md += ["![segment lengths](figures/segment_lengths.png)", ""]

    # ---- embeddings: geometry ------------------------------------------------------------
    print("embedding geometry", flush=True)
    md += ["## 5. Embeddings", "",
           "These are computed on segment mean-pooled vectors (float16 stored, float32 here). **Anisotropy** is the mean "
           "pairwise cosine similarity, where 0 means isotropic. **PCA-90** is the number of principal components that "
           "explain 90% of the variance. **Between-session share** is the fraction of total variance explained by "
           "session means, so a high value means the layer mostly encodes who is speaking and the recording. "
           "**Duration R²** is a ridge regression from the layer to log segment length, with session-grouped 5-fold CV. "
           "**Gender AUC** is a session-level 5-fold CV probe.", ""]
    sess_arr = seg.session_id.values
    logdur = np.log(seg.dur_s.values)
    geo: dict = {}
    vecs: dict = {}
    for m in ("whisper_small", "hubert_base", "roberta_base"):
        geo[m] = []
        for k in range(13):
            X = ds.pooled(m, k) if m != "roberta_base" else ds.text("roberta_base", k)
            g = layer_geometry(X, sess_arr)
            g["duration_r2"] = duration_r2(X, logdur, sess_arr)
            Xs, _, _ = ds.session_pool(X, weight_by_duration=True)
            g["gender_auc"] = gender_auc(Xs, lab.gender.values.astype(int))
            geo[m].append(g)
            vecs[(m, k)] = X
        print(f"  {m} done", flush=True)
    others = {"mpnet": ds.text("mpnet"), "egemaps": ds.egemaps().values.astype(np.float32)}
    for name, X in others.items():
        g = layer_geometry(X, sess_arr, standardize=(name == "egemaps"))
        Xd = np.nan_to_num(X)
        g["duration_r2"] = duration_r2(Xd, logdur, sess_arr)
        g["gender_auc"] = gender_auc(ds.session_pool(Xd, weight_by_duration=True)[0], lab.gender.values.astype(int))
        geo[name] = [g]
    R["embedding_geometry"] = geo
    T["geometry"] = [{"model": m, "layer": (k if m in MODEL_NAMES else None),
                      **{c: (None if isinstance(v, float) and np.isnan(v) else round(float(v), 4)) for c, v in g.items()}}
                     for m in geo for k, g in enumerate(geo[m])]
    keys = [("between_session_share", "between-session share"), ("anisotropy", "anisotropy"), ("pca90", "PCA-90"),
            ("duration_r2", "duration R²"), ("gender_auc", "gender AUC"), ("norm", "mean norm")]
    for m in ("whisper_small", "hubert_base", "roberta_base"):
        md += [f"**{MODEL_NAMES[m]}**", "",
               md_table(["layer"] + [nm for _, nm in keys],
                        [[k] + [fmt(geo[m][k][c], 0 if c == "pca90" else 2) for c, _ in keys] for k in range(13)]), ""]
    md += [md_table(["features"] + [nm for _, nm in keys],
                    [[n] + [fmt(geo[n][0][c], 0 if c == "pca90" else 2) for c, _ in keys] for n in others]),
           "", "eGeMAPS is z-scored per feature before the geometry measures.", ""]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (c, nm) in zip(axes, keys[:4]):
        for m in ("whisper_small", "hubert_base", "roberta_base"):
            ax.plot(range(13), [g[c] for g in geo[m]], color=MODEL_COLORS[m], marker="o", ms=3, label=MODEL_NAMES[m])
        ax.set(title=nm, xlabel="layer")
        ax.set_xticks([0, 3, 6, 9, 12])
    axes[0].legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(figs / "layer_geometry.png", dpi=150)
    plt.close(fig)
    md += ["![layer geometry](figures/layer_geometry.png)", ""]

    # ---- CKA --------------------------------------------------------------------------
    print("CKA", flush=True)
    sub = rng.choice(len(seg), min(4000, len(seg)), replace=False)
    names = [(m, k) for m in ("whisper_small", "hubert_base", "roberta_base") for k in range(13)]
    C = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            C[i, j] = C[j, i] = cka(vecs[names[i]][sub].astype(np.float64), vecs[names[j]][sub].astype(np.float64))
    R["cka"] = {"order": [f"{m}:{k}" for m, k in names], "matrix": C.round(4).tolist()}
    T["cka"] = [{"model_a": names[i][0], "layer_a": names[i][1], "model_b": names[j][0], "layer_b": names[j][1],
                 "cka": round(float(C[i, j]), 4)} for i in range(len(names)) for j in range(len(names))]
    best = {}
    for a, b in [("whisper_small", "hubert_base"), ("whisper_small", "roberta_base"), ("hubert_base", "roberta_base")]:
        ia = [i for i, n in enumerate(names) if n[0] == a]
        ib = [i for i, n in enumerate(names) if n[0] == b]
        blk = C[np.ix_(ia, ib)]
        i, j = np.unravel_index(blk.argmax(), blk.shape)
        best[f"{a}~{b}"] = {"max": float(blk.max()), "layers": [int(i), int(j)]}
    R["cka_best"] = best
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(C, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    for t in (12.5, 25.5):
        ax.axhline(t, color=SURFACE, lw=2)
        ax.axvline(t, color=SURFACE, lw=2)
    ticks = [6, 19, 32]
    ax.set_xticks(ticks, ["Whisper", "HuBERT", "RoBERTa"])
    ax.set_yticks(ticks, ["Whisper", "HuBERT", "RoBERTa"])
    ax.grid(False)
    ax.set_title("Linear CKA between layers (0→12 within each block)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figs / "cka.png", dpi=150)
    plt.close(fig)
    md += ["### Representational similarity (linear CKA)", "",
           f"This is computed on {len(sub):,} random segments. The most similar layer pair for each model pair is: " + "; ".join(
               f"{MODEL_NAMES[k.split('~')[0]]} L{v['layers'][0]} ~ {MODEL_NAMES[k.split('~')[1]]} L{v['layers'][1]}: "
               f"{v['max']:.2f}" for k, v in best.items()) + ".", "", "![CKA](figures/cka.png)", ""]

    # ---- probes --------------------------------------------------------------------------
    print("probes", flush=True)
    md += ["### Session-level PHQ-8 probes", "",
           "- **Features:** segment vectors are averaged into one vector per session, weighted by duration.",
           "- **Models:** logistic regression for PHQ-8 ≥ 10 (AUC) and ridge regression for the PHQ-8 score (RMSE). "
           "Hyperparameters are chosen by 5-fold CV within the training sessions.",
           "- **Evaluation:** train on train and score on dev; train on train + dev and score on test.",
           "- **Uncertainty:** the 95% intervals are a bootstrap over test sessions. With only 32 to 45 evaluation "
           "sessions they are wide, so read the curves as a baseline, not a result.", ""]
    probes: dict = {}
    if lab.split.nunique() == 3 and lab.groupby("split").phq8_binary.nunique().min() == 2 and len(lab) >= 30:
        for m in ("whisper_small", "hubert_base", "roberta_base"):
            probes[m] = [session_probe(ds.session_pool(vecs[(m, k)], weight_by_duration=True)[0], lab, rng)
                         for k in range(13)]
            print(f"  {m} done", flush=True)
        for n, X in others.items():
            probes[n] = [session_probe(ds.session_pool(np.nan_to_num(X), weight_by_duration=True)[0], lab, rng)]
        R["probes"] = probes
        T["probes"] = [{"model": m, "layer": (k if m in MODEL_NAMES else None),
                        "auc_dev": round(p["dev"]["auc"], 4), "auc_test": round(p["test"]["auc"], 4),
                        "auc_test_lo": round(p["test"]["auc_lo"], 4), "auc_test_hi": round(p["test"]["auc_hi"], 4),
                        "rmse_test": round(p["test"]["rmse"], 3), "rmse_test_mean_only": round(p["test"]["rmse_baseline"], 3),
                        "rmse_dev": round(p["dev"]["rmse"], 3), "rmse_dev_mean_only": round(p["dev"]["rmse_baseline"], 3)}
                       for m in probes for k, p in enumerate(probes[m])]
        rows = []
        for m in ("whisper_small", "hubert_base", "roberta_base"):
            for k in range(13):
                p = probes[m][k]
                rows.append([MODEL_NAMES[m], k, fmt(p["dev"]["auc"]), fmt(p["test"]["auc"]),
                             f"{fmt(p['test']['auc_lo'])}–{fmt(p['test']['auc_hi'])}",
                             fmt(p["test"]["rmse"]), fmt(p["test"]["rmse_baseline"])])
        for n in others:
            p = probes[n][0]
            rows.append([n, "–", fmt(p["dev"]["auc"]), fmt(p["test"]["auc"]),
                         f"{fmt(p['test']['auc_lo'])}–{fmt(p['test']['auc_hi'])}",
                         fmt(p["test"]["rmse"]), fmt(p["test"]["rmse_baseline"])])
        md += [md_table(["features", "layer", "AUC dev", "AUC test", "test 95% CI", "RMSE test", "RMSE mean-only"], rows), ""]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
        for ax, split in zip(axes, ["dev", "test"]):
            for m in ("whisper_small", "hubert_base", "roberta_base"):
                ax.plot(range(13), [p[split]["auc"] for p in probes[m]], color=MODEL_COLORS[m], marker="o", ms=3,
                        label=MODEL_NAMES[m])
            for n, ls in [("mpnet", ":"), ("egemaps", "--")]:
                v = probes[n][0][split]["auc"]
                ax.axhline(v, color=INK2, lw=1, ls=ls)
                ax.text(12.3, v, n, color=INK2, fontsize=7.5, va="center")
            ax.axhline(0.5, color=GRID, lw=1)
            ax.set(title=f"PHQ-8 ≥ 10 AUC, {split}", xlabel="layer")
            ax.set_xticks([0, 3, 6, 9, 12])
            ax.set_xlim(-0.5, 14)
        axes[0].legend(fontsize=7.5, loc="lower left")
        fig.tight_layout()
        fig.savefig(figs / "probes.png", dpi=150)
        plt.close(fig)
        md += ["![probes](figures/probes.png)", ""]
    else:
        md += ["*Skipped: too few sessions per split for a session-level probe.*", ""]

    # ---- eGeMAPS by group --------------------------------------------------------------------
    eg = ds.egemaps()
    pick = {"F0semitoneFrom27.5Hz_sma3nz_amean": "F0 mean (semitones)",
            "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "F0 variability",
            "loudness_sma3_amean": "loudness mean", "jitterLocal_sma3nz_amean": "jitter",
            "shimmerLocaldB_sma3nz_amean": "shimmer (dB)", "HNRdBACF_sma3nz_amean": "HNR (dB)",
            "VoicedSegmentsPerSec": "voiced segments per s", "MeanUnvoicedSegmentLength": "mean unvoiced segment (s)",
            "equivalentSoundLevel_dBp": "equivalent sound level (dB)"}
    egs = eg[[c for c in pick if c in eg]].copy()
    egs["session_id"] = seg.session_id.values
    egs["w"] = seg.dur_s.values
    sess_eg = egs.groupby("session_id").apply(
        lambda d: pd.Series({c: np.average(d[c], weights=d.w) for c in pick if c in d}), include_groups=False)
    md += ["### eGeMAPS by group (per-session duration-weighted means)", ""]
    group_table(sess_eg, {c: pick[c] for c in sess_eg.columns}, "egemaps_by_group")

    # ---- frames --------------------------------------------------------------------------------
    print("frames", flush=True)
    from safetensors.numpy import load_file

    fr_rows = []
    for s in sessions[: args.frame_sessions]:
        for m in ("whisper_small", "hubert_base"):
            p = files.get(f"{m}/frames/{s}.safetensors")
            if p is None:
                continue
            d = load_file(p)
            for k in ds.frame_layers():
                arrs = [v.astype(np.float32) for key, v in d.items() if key.endswith(f"layer_{k:02d}")]
                norms = np.concatenate([np.linalg.norm(a, axis=1) for a in arrs])
                cos = np.concatenate([
                    (a[1:] * a[:-1]).sum(1) / (np.linalg.norm(a[1:], axis=1) * np.linalg.norm(a[:-1], axis=1) + 1e-9)
                    for a in arrs if len(a) > 1])
                fr_rows.append({"model": m, "layer": k, "session": s, "frames": int(sum(len(a) for a in arrs)),
                                "norm": float(norms.mean()), "adjacent_cos": float(cos.mean())})
            if files.root is None or not (files.root / f"{m}/frames/{s}.safetensors").exists():
                try:
                    p.unlink()
                except OSError:
                    pass
    if fr_rows:
        fr = pd.DataFrame(fr_rows).groupby(["model", "layer"]).agg(
            sessions=("session", "nunique"), frames=("frames", "sum"), norm=("norm", "mean"),
            adjacent_cos=("adjacent_cos", "mean")).reset_index()
        R["frames"] = fr.to_dict("records")
        T["frames"] = [{"model": r.model, "layer": int(r.layer), "sessions": int(r.sessions), "frames": int(r.frames),
                        "mean_norm": round(float(r.norm), 3), "adjacent_cos": round(float(r.adjacent_cos), 4)}
                       for r in fr.itertuples()]
        md += ["### Frame-level outputs", "",
               f"These are from the first {fr.sessions.max()} sessions. **Adjacent cos** is the mean cosine similarity "
               "between consecutive 20 ms frames, a measure of how smooth each layer is over time.", "",
               md_table(["model", "layer", "frames", "mean norm", "adjacent cos"],
                        [[MODEL_NAMES[r.model], r.layer, fmt(int(r.frames)), fmt(r.norm), fmt(r.adjacent_cos)]
                         for r in fr.itertuples()]), ""]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "STATS.md").write_text("\n".join(md) + "\n")
    (args.out / "stats.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    ddir = args.out / "data"
    ddir.mkdir(exist_ok=True)
    code = Path(__file__).read_text()
    src = (f"{args.hf or args.root} at {args.revision or 'local'}; dataset version {man.get('dataset_version')}; "
           f"code {man['code_commit'][:7]}")
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        if hasattr(v, "item"):
            v = v.item()
        return None if isinstance(v, float) and not np.isfinite(v) else v

    for name, rows_ in T.items():
        (ddir / f"{name}.json").write_text(json.dumps({"rows": clean(rows_), "meta": [
            {"label": "analyze.py", "value": code, "type": "code", "lang": "python"},
            {"label": "Input", "value": src}]}, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {args.out}/STATS.md and {len(T)} tables in {ddir}")


if __name__ == "__main__":
    main()
