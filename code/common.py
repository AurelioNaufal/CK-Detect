"""Shared infrastructure: paths, data loading, splits, compression, metrics, timing."""
from __future__ import annotations

import gzip
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
METHOD = ROOT / "Method"
DATASETS = ROOT / "Datasets"
CACHE = METHOD / "cache"
SPLITS_DIR = CACHE / "splits"
FEATURES_DIR = CACHE / "features"
PREDICTIONS_DIR = CACHE / "predictions"
RESULTS_DIR = METHOD / "results"

for d in (CACHE, SPLITS_DIR, FEATURES_DIR, PREDICTIONS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Dataset file mapping
# ------------------------------------------------------------------ #
# rewrite_file: per-class json with `input` field = original + multiple prompt-keys = rewrites
# Note: HC3 has no GPT-rewriter file (use 'qwen' for HC3).
DATASET_FILES: Dict[str, Dict[str, Dict[str, str]]] = {
    "yelp": {
        "gpt":  {"ai": "Yelp/rewrite_yelp_GPT_usingGPT.json",
                 "human": "Yelp/rewrite_yelp_human_usingGPT.json"},
        "qwen": {"ai": "Yelp/rewrite_yelp_GPT_usingqwen.json",
                 "human": "Yelp/rewrite_yelp_human_usingqwen.json"},
        "qwen08b": {"ai": "Yelp/rewrite_yelp_gpt_usingqwen08b.json",
                    "human": "Yelp/rewrite_yelp_human_usingqwen08b.json"},
    },
    "arxiv": {
        "gpt":  {"ai": "Arxiv/rewrite_arxiv_GPT_usinggpt.json",
                 "human": "Arxiv/rewrite_arxiv_human_usinggpt.json"},
        "qwen": {"ai": "Arxiv/rewrite_arxiv_GPT_usingqwen.json",
                 "human": "Arxiv/rewrite_arxiv_human_usingqwen.json"},
        "qwen08b": {"ai": "Arxiv/rewrite_arxiv_GPT_usingqwen08b.json",
                    "human": "Arxiv/rewrite_arxiv_human_usingqwen08b.json"},
    },
    "code": {
        "gpt":  {"ai": "Code/rewrite_code_GPT_usingGPT.json",
                 "human": "Code/rewrite_code_human_usingGPT.json"},
        "qwen": {"ai": "Code/rewrite_code_GPT_usingqwen.json",
                 "human": "Code/rewrite_code_human_usingqwen.json"},
        "qwen08b": {"ai": "Code/rewrite_code_GPT_usingqwen08b.json",
                    "human": "Code/rewrite_code_human_usingqwen08b.json"},
    },
    "essay": {
        "gpt":  {"ai": "Essay/essay_gpt.json",
                 "human": "Essay/essay_human.json"},
        "qwen": {"ai": "Essay/qwen_rewrite_essay_gpt.json",
                 "human": "Essay/qwen_rewrite_essay_human.json"},
        "qwen08b": {"ai": "Essay/qwen08b_rewrite_essay_gpt.json",
                    "human": "Essay/qwen08b_rewrite_essay_human.json"},
    },
    "cwriting": {
        "gpt":  {"ai": "CWriting/GPT.json",
                 "human": "CWriting/Human.json"},
        "qwen": {"ai": "CWriting/qwen_rewrite_cwriting_gpt.json",
                 "human": "CWriting/qwen_rewrite_cwriting_human.json"},
        "qwen08b": {"ai": "CWriting/qwen08b_rewrite_cwriting_gpt.json",
                    "human": "CWriting/qwen08b_rewrite_cwriting_human.json"},
    },
    "news": {
        "gpt":  {"ai": "News/GPT.json",
                 "human": "News/Human.json"},
        "qwen": {"ai": "News/qwen_rewrite_news_gpt.json",
                 "human": "News/qwen_rewrite_news_human.json"},
        "qwen08b": {"ai": "News/qwen08b_rewrite_news_gpt.json",
                    "human": "News/qwen08b_rewrite_news_human.json"},
    },
}

# Per-domain hard cap on (per-class) samples after balancing
PER_CLASS_CAP: Dict[str, int] = {
    "yelp": 2000, "arxiv": 2000, "code": 2000, "essay": 2000,
    "cwriting": None, "news": None,
}

# HC3 is intentionally excluded: no GPT rewrites available and removed from
# the canonical sweep entirely.
ALL_DOMAINS = ["yelp", "arxiv", "code", "essay", "cwriting", "news"]
MAIN_DOMAINS = ALL_DOMAINS   # alias kept for back-compat
REWRITERS = ["gpt", "qwen", "qwen08b"]

DEFAULT_REWRITER: Dict[str, str] = {
    "yelp": "gpt", "arxiv": "gpt", "code": "gpt", "essay": "gpt",
    "cwriting": "gpt", "news": "gpt",
}

# Domain - rewriter availability
def has_rewriter(domain: str, rewriter: str) -> bool:
    return rewriter in DATASET_FILES[domain]

# ------------------------------------------------------------------ #
# Seeding
# ------------------------------------------------------------------ #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #
def _coerce_text(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        return "\n".join(str(x) for x in item)
    if isinstance(item, dict):
        for k in ("input", "text", "abs", "content"):
            if k in item:
                return _coerce_text(item[k])
    return str(item)


def load_rewrite_file(rel_path: str) -> List[Dict]:
    """Load a rewrite-style json (list of dicts with `input` + prompt-keys).
    For raw list-of-string files (Arxiv/HC3/Yelp ai_path edge cases), wraps each
    string into a dict with only `input`.
    """
    full = DATASETS / rel_path
    with open(full, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data:
        if isinstance(item, dict):
            d = dict(item)
            if "input" not in d:
                # Arxiv 'abs' style
                if "abs" in d:
                    d["input"] = d["abs"]
                else:
                    d["input"] = next(iter(d.values()))
            d["input"] = _coerce_text(d["input"])
            for k, v in list(d.items()):
                if k != "input":
                    d[k] = _coerce_text(v)
            out.append(d)
        else:
            out.append({"input": _coerce_text(item)})
    return out


def get_prompt_keys(records: List[Dict]) -> List[str]:
    """Return the list of rewrite-prompt keys (all keys != 'input', common_features, etc.)."""
    skip = {"input", "common_features", "fzwz_features",
            "avg_common_features", "common_features_ori_vs_allcombined",
            "title", "abs"}
    if not records:
        return []
    return [k for k in records[0].keys() if k not in skip]


# ------------------------------------------------------------------ #
# Train/test split (shared across all methods)
# ------------------------------------------------------------------ #
def make_split(domain: str, rewriter: str = "qwen", overwrite: bool = False) -> Dict:
    """
    Build (or load cached) train/test split for a (domain, rewriter).
    The split is on INDICES into the rewrite file (so all methods can map back
    consistently). Stratified 80/20, seed=42. Per-class cap applied if set.
    Returns dict with keys: domain, rewriter, ai_path, human_path,
        train_idx_ai, train_idx_human, test_idx_ai, test_idx_human, prompt_keys.
    """
    cache_file = SPLITS_DIR / f"{domain}_{rewriter}.json"
    if cache_file.exists() and not overwrite:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if not has_rewriter(domain, rewriter):
        raise ValueError(f"No {rewriter} rewriter for domain {domain}")

    cfg = DATASET_FILES[domain][rewriter]
    ai = load_rewrite_file(cfg["ai"])
    hu = load_rewrite_file(cfg["human"])

    cap = PER_CLASS_CAP.get(domain)
    n_ai = len(ai) if cap is None else min(len(ai), cap)
    n_hu = len(hu) if cap is None else min(len(hu), cap)

    rng = random.Random(SEED)
    ai_idx = list(range(len(ai))); rng.shuffle(ai_idx); ai_idx = sorted(ai_idx[:n_ai])
    hu_idx = list(range(len(hu))); rng.shuffle(hu_idx); hu_idx = sorted(hu_idx[:n_hu])

    # Stratified 80/20 within each class
    tr_ai, te_ai = train_test_split(ai_idx, test_size=0.2, random_state=SEED)
    tr_hu, te_hu = train_test_split(hu_idx, test_size=0.2, random_state=SEED)

    split = {
        "domain": domain,
        "rewriter": rewriter,
        "ai_path": cfg["ai"],
        "human_path": cfg["human"],
        "n_ai_total": len(ai),
        "n_human_total": len(hu),
        "n_ai_used": n_ai,
        "n_human_used": n_hu,
        "train_idx_ai": sorted(tr_ai),
        "test_idx_ai":  sorted(te_ai),
        "train_idx_human": sorted(tr_hu),
        "test_idx_human":  sorted(te_hu),
        "prompt_keys": get_prompt_keys(ai),
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


# ------------------------------------------------------------------ #
# Compression (gzip level 9)
# ------------------------------------------------------------------ #
def gz(s: str) -> int:
    return len(gzip.compress(s.encode("utf-8"), compresslevel=9))

# ------------------------------------------------------------------ #
# Metrics + LR helper
# ------------------------------------------------------------------ #
def evaluate(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray = None,
             ms_per_text: float = None) -> Dict:
    """Report Accuracy, macro-F1 (handles class imbalance fairly), and AUC."""
    if y_pred is None:
        y_pred = (y_score > np.median(y_score)).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc":      float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0,
    }
    if ms_per_text is not None:
        metrics["ms_per_text"] = float(ms_per_text)
    return metrics


def save_predictions(method: str, domain: str, rewriter: str,
                     y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Path:
    """Cache per-test predictions for later analysis (confusion matrices,
    threshold sweeps, error analysis)."""
    out = PREDICTIONS_DIR / f"{method}_{domain}_{rewriter}.npz"
    np.savez_compressed(out,
                        y_true=np.asarray(y_true, dtype=np.int8),
                        y_score=np.asarray(y_score, dtype=np.float64),
                        y_pred=np.asarray(y_pred, dtype=np.int8))
    return out


def fit_logreg(X_train, y_train, X_test, y_test, ms_per_text=None,
               cache_key: tuple = None) -> Dict:
    """Fit a StandardScaler -> LogisticRegression pipeline and return metrics.
    Also records `train_time_s` = classifier fit wall-time (excludes feature
    extraction, which is captured separately as ms_per_text * n_train)."""
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0
    y_score = pipe.predict_proba(X_test)[:, 1]
    y_pred  = pipe.predict(X_test)
    if cache_key is not None:
        save_predictions(*cache_key, y_test, y_score, y_pred)
    metrics = evaluate(y_test, y_score, y_pred, ms_per_text)
    metrics["train_time_s"] = float(train_time_s)
    return metrics


def fit_threshold(scores_train, y_train, scores_test, y_test,
                  higher_is_ai: bool = True, ms_per_text: float = None,
                  cache_key: tuple = None) -> Dict:
    """Fit a 1-D logistic regression on a single score; record fit time
    in train_time_s (typically < 1 ms)."""
    X_tr = np.array(scores_train, dtype=np.float64).reshape(-1, 1)
    X_te = np.array(scores_test,  dtype=np.float64).reshape(-1, 1)
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_train)
    train_time_s = time.perf_counter() - t0
    y_score = clf.predict_proba(X_te)[:, 1]
    y_pred  = clf.predict(X_te)
    if cache_key is not None:
        save_predictions(*cache_key, y_test, y_score, y_pred)
    metrics = evaluate(y_test, y_score, y_pred, ms_per_text)
    metrics["train_time_s"] = float(train_time_s)
    return metrics


# ------------------------------------------------------------------ #
# Timing helper
# ------------------------------------------------------------------ #
class Timer:
    def __init__(self):
        self.t = None
    def __enter__(self):
        self.t = time.perf_counter()
        return self
    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t


# ------------------------------------------------------------------ #
# Result serialization
# ------------------------------------------------------------------ #
def save_result(method: str, domain: str, rewriter: str, payload: Dict) -> Path:
    """Save method results JSON to results/<method>/<domain>_<rewriter>.json."""
    out_dir = RESULTS_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{domain}_{rewriter}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out
