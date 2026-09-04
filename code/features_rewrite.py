"""Per-sample rewrite-conditioned features.

For each (x, x'_n) pair we compute 6 scalar features:
    cond_raw   = g(x x') - g(x')                # Eq. cond_raw
    cond_norm  = (g(x x') - g(x')) / g(x)       # Eq. cond_norm
    ncd        = (g(x x') - min(g(x),g(x'))) / max(g(x),g(x'))
    g_rewrite  = g(x')
    bag_ngram  = mean( sum_{k=1..4} |common k-grams(x,x')| ) / len_x
    leven      = fuzz.token_set_ratio(x, x') / 100.0

We compute these for every available prompt, then aggregate across the N
prompts with mean / std / min / max -> 24-dim feature vector per sample.

Per-sample features are cached as a single .npz so methods can reuse them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from common import (CACHE, FEATURES_DIR, Timer, gz, load_rewrite_file, make_split,
                    set_seed)

try:
    from fuzzywuzzy import fuzz
    _HAVE_FUZZ = True
except ImportError:
    _HAVE_FUZZ = False


# ------------------------------------------------------------------ #
# Surface-similarity helpers (mirrors Raidar)
# ------------------------------------------------------------------ #
def _tok(s: str) -> List[str]:
    return [w.lower().strip() for w in s.split()]

def _ngrams(toks: List[str], n: int) -> List[str]:
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]

def _bag_score(original: str, rewrite: str, n_max: int = 4) -> float:
    """Sum over n=1..n_max of |common n-grams|, divided by len(tokens(original))."""
    a = _tok(original); b = _tok(rewrite)
    if not a:
        return 0.0
    total = 0
    for n in range(1, n_max + 1):
        sa = set(_ngrams(a, n)) if n > 1 else set(a)
        sb = set(_ngrams(b, n)) if n > 1 else set(b)
        total += len(sa & sb)
    return total / len(a)

def _leven_score(original: str, rewrite: str) -> float:
    if _HAVE_FUZZ:
        return fuzz.token_set_ratio(original, rewrite) / 100.0
    # Fallback: Jaccard on tokens
    a, b = set(_tok(original)), set(_tok(rewrite))
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# ------------------------------------------------------------------ #
# Per-(x, x') feature row
# ------------------------------------------------------------------ #
def per_pair_features(x: str, x_prime: str, g_x: int = None) -> np.ndarray:
    if g_x is None:
        g_x = gz(x)
    g_xp = gz(x_prime)
    g_xx = gz(x + x_prime)
    cond_raw = float(g_xx - g_xp)
    cond_norm = cond_raw / max(1, g_x)
    ncd = (g_xx - min(g_x, g_xp)) / max(1, max(g_x, g_xp))
    return np.array([cond_raw, cond_norm, ncd, float(g_xp),
                     _bag_score(x, x_prime), _leven_score(x, x_prime)],
                    dtype=np.float64)


FEATURE_NAMES = ["cond_raw", "cond_norm", "ncd", "g_rewrite", "bag_ngram", "leven"]
AGG_NAMES = ["mean", "std", "min", "max"]


def aggregate(per_prompt: np.ndarray) -> np.ndarray:
    """per_prompt: (N, 6) -> (24,) [mean, std, min, max for each feature]."""
    if per_prompt.size == 0:
        return np.zeros(24, dtype=np.float64)
    return np.concatenate([
        per_prompt.mean(axis=0), per_prompt.std(axis=0),
        per_prompt.min(axis=0),  per_prompt.max(axis=0),
    ]).astype(np.float64)


# ------------------------------------------------------------------ #
# Per-domain feature computation (with disk cache)
# ------------------------------------------------------------------ #
def compute_features(domain: str, rewriter: str, overwrite: bool = False) -> Dict:
    """
    Compute per-sample 24-d feature matrix + per-sample inference time.
    Caches to cache/features/rewrite_<domain>_<rewriter>.npz.

    Returns dict with keys:
      X_ai, X_human: (n, 24)
      g_orig_ai, g_orig_human: (n,)  -- g(x) for each sample (handy for other methods)
      per_prompt_ai, per_prompt_human: (n, N, 6)  for ablation / single-prompt analysis
      ms_per_text:  mean wall-clock per sample
      prompt_keys, ai_indices, human_indices
    """
    set_seed()
    cache_file = FEATURES_DIR / f"rewrite_{domain}_{rewriter}.npz"
    if cache_file.exists() and not overwrite:
        return _load_npz(cache_file)

    split = make_split(domain, rewriter)
    ai_records = load_rewrite_file(split["ai_path"])
    hu_records = load_rewrite_file(split["human_path"])
    prompts    = split["prompt_keys"]

    ai_idx = sorted(split["train_idx_ai"] + split["test_idx_ai"])
    hu_idx = sorted(split["train_idx_human"] + split["test_idx_human"])

    def _process(records, idxs):
        X = np.zeros((len(idxs), 24), dtype=np.float64)
        per_pp = np.zeros((len(idxs), len(prompts), 6), dtype=np.float64)
        g_orig = np.zeros(len(idxs), dtype=np.float64)
        t_total = 0.0
        for i, ix in enumerate(idxs):
            rec = records[ix]
            x = rec["input"]
            g_x = gz(x)
            g_orig[i] = g_x
            with Timer() as t:
                rows = []
                for j, p in enumerate(prompts):
                    xp = rec.get(p)
                    if xp is None or not isinstance(xp, str) or not xp.strip():
                        # missing rewrite -> use original as a neutral fallback
                        xp = x
                    f = per_pair_features(x, xp, g_x)
                    per_pp[i, j] = f
                    rows.append(f)
                X[i] = aggregate(np.vstack(rows))
            t_total += t.elapsed
        return X, per_pp, g_orig, t_total

    print(f"  [rewrite-feat] {domain}/{rewriter}: AI ({len(ai_idx)}) ...", flush=True)
    X_ai, pp_ai, g_ai, t_ai = _process(ai_records, ai_idx)
    print(f"  [rewrite-feat] {domain}/{rewriter}: Human ({len(hu_idx)}) ...", flush=True)
    X_hu, pp_hu, g_hu, t_hu = _process(hu_records, hu_idx)
    ms_per_text = 1000.0 * (t_ai + t_hu) / (len(ai_idx) + len(hu_idx))

    payload = {
        "X_ai": X_ai, "X_human": X_hu,
        "per_prompt_ai": pp_ai, "per_prompt_human": pp_hu,
        "g_orig_ai": g_ai, "g_orig_human": g_hu,
        "ai_indices": np.array(ai_idx, dtype=np.int64),
        "human_indices": np.array(hu_idx, dtype=np.int64),
        "ms_per_text": np.array([ms_per_text]),
        "prompt_keys": np.array(prompts),
        "feature_names": np.array([f"{a}_{f}" for a in AGG_NAMES for f in FEATURE_NAMES]),
    }
    np.savez_compressed(cache_file, **payload)
    print(f"  [rewrite-feat] saved -> {cache_file}  ({ms_per_text:.2f} ms/text)")
    return _load_npz(cache_file)


def _load_npz(path: Path) -> Dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def build_xy_for_split(features: Dict, split: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Slice X_ai/X_human by the saved train/test indices and return
    (X_train, y_train, X_test, y_test, ms_per_text)."""
    ai_idx = features["ai_indices"].tolist()
    hu_idx = features["human_indices"].tolist()

    def pos(idx_list, target):  # map global index -> row in X
        return [idx_list.index(t) for t in target]

    tr_ai = pos(ai_idx, split["train_idx_ai"])
    te_ai = pos(ai_idx, split["test_idx_ai"])
    tr_hu = pos(hu_idx, split["train_idx_human"])
    te_hu = pos(hu_idx, split["test_idx_human"])

    X_tr = np.vstack([features["X_ai"][tr_ai], features["X_human"][tr_hu]])
    X_te = np.vstack([features["X_ai"][te_ai], features["X_human"][te_hu]])
    y_tr = np.array([1]*len(tr_ai) + [0]*len(tr_hu))
    y_te = np.array([1]*len(te_ai) + [0]*len(te_hu))
    ms = float(features["ms_per_text"][0])
    return X_tr, y_tr, X_te, y_te, ms
