"""Method 2: Corpus-Conditioned Kolmogorov Complexity.

Build class-specific reference corpora D_AI / D_H from TRAINING samples only,
then compute g(x | D) approximations for every sample (train and test).

NO DATA LEAKAGE: D_AI and D_H contain only training-split originals.

Features per sample (5-d):
    g_x_given_ai, g_x_given_human, M, M_raw, g_x

Implementation: stateful zlib compressor (`zlib.compressobj(level=9, wbits=31)`
= gzip wire format) is fed the corpus ONCE during training; that state is
cloned per test sample and only the marginal SEP+x bytes are compressed.
This makes per-sample inference cost independent of corpus size (~1 ms).
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from common import (FEATURES_DIR, Timer, gz, load_rewrite_file, make_split,
                    set_seed)

SEPARATOR = "\n\n"
SEP_B = SEPARATOR.encode("utf-8")
# wbits = 31 → max-window deflate + gzip header/trailer ≡ gzip.compress format,
# minimising numerical drift from the prior gzip.compress-based pipeline.
_WBITS = 31
_LEVEL = 9


def _build_db(records, idxs) -> str:
    return SEPARATOR.join(records[i]["input"] for i in idxs)


def fit_corpus_compressor(corpus_text: str) -> tuple:
    """Train-time: feed the full corpus into a streaming gzip encoder and
    return (compressor_with_state, g_D) where g_D is bytes-when-flushed.
    The returned compressor is NOT flushed; later `.copy()` is used to score
    marginal compressed sizes per test sample."""
    co = zlib.compressobj(_LEVEL, zlib.DEFLATED, _WBITS)
    prefix = co.compress(corpus_text.encode("utf-8"))
    # Compute g(D) by flushing a copy (does not disturb co's state).
    g_D = len(prefix) + len(co.copy().flush())
    return co, prefix, g_D


def conditional_g(co_state: zlib.compressobj, prefix: bytes, x: str) -> int:
    """Inference-time: g(D + SEP + x) using saved compressor state.
    `co_state` is the trained corpus compressor; we clone its state and feed
    only the marginal bytes (SEP + x), then flush."""
    co = co_state.copy()
    tail = co.compress(SEP_B + x.encode("utf-8")) + co.flush()
    return len(prefix) + len(tail)


def compute_features(domain: str, rewriter: str, overwrite: bool = False) -> Dict:
    """
    Returns dict (also cached as .npz):
        M_ai, M_human:       (n_ai,), (n_hu,)  -- the M(x) score
        g_ai_given_ai, g_ai_given_human, ...   each (n_ai,) or (n_hu,)
        g_x_ai, g_x_human
        ai_indices, human_indices              -- global indices into the file
        ms_per_text:                           -- mean test-time per sample (excludes 1x dict prep)
    """
    set_seed()
    cache_file = FEATURES_DIR / f"corpus_{domain}_{rewriter}.npz"
    if cache_file.exists() and not overwrite:
        z = np.load(cache_file, allow_pickle=True)
        return {k: z[k] for k in z.files}

    split = make_split(domain, rewriter)
    ai_records = load_rewrite_file(split["ai_path"])
    hu_records = load_rewrite_file(split["human_path"])

    tr_ai_idx = split["train_idx_ai"]
    tr_hu_idx = split["train_idx_human"]

    # ---- Train: build corpora + fit stateful compressors (ONE-TIME) ----
    print(f"  [corpus-feat] {domain}/{rewriter}: building D_AI from {len(tr_ai_idx)} texts "
          f"and D_H from {len(tr_hu_idx)} texts ...", flush=True)
    import time as _t
    _t0 = _t.perf_counter()
    D_ai = _build_db(ai_records, tr_ai_idx)
    D_hu = _build_db(hu_records, tr_hu_idx)
    co_ai, prefix_ai, g_D_ai = fit_corpus_compressor(D_ai)
    co_hu, prefix_hu, g_D_hu = fit_corpus_compressor(D_hu)
    train_build_s = _t.perf_counter() - _t0
    print(f"  [corpus-feat] g(D_AI)={g_D_ai:,}  g(D_H)={g_D_hu:,}  build={train_build_s:.2f}s",
          flush=True)

    def _score(records, idxs):
        n = len(idxs)
        g_ai  = np.zeros(n, dtype=np.float64)
        g_hu  = np.zeros(n, dtype=np.float64)
        g_x   = np.zeros(n, dtype=np.float64)
        M     = np.zeros(n, dtype=np.float64)
        M_raw = np.zeros(n, dtype=np.float64)
        t_total = 0.0
        for i, ix in enumerate(idxs):
            x = records[ix]["input"]
            with Timer() as t:
                # Inference: pay only the marginal cost of compressing x given
                # the pre-trained corpus state. gz(x) is for the feature vector.
                g_x[i]  = gz(x)
                g_ai[i] = conditional_g(co_ai, prefix_ai, x) - g_D_ai
                g_hu[i] = conditional_g(co_hu, prefix_hu, x) - g_D_hu
                denom = g_ai[i] + g_hu[i]
                M[i] = (g_ai[i] - g_hu[i]) / denom if denom != 0 else 0.0
                M_raw[i] = g_ai[i] - g_hu[i]
            t_total += t.elapsed
            if (i + 1) % 200 == 0:
                print(f"    [corpus-feat] scored {i+1}/{n}", flush=True)
        return g_ai, g_hu, g_x, M, M_raw, t_total

    # All ai+human samples (train+test)
    ai_idx_all = sorted(split["train_idx_ai"] + split["test_idx_ai"])
    hu_idx_all = sorted(split["train_idx_human"] + split["test_idx_human"])

    print(f"  [corpus-feat] scoring {len(ai_idx_all)} AI samples ...", flush=True)
    g_ai_a, g_hu_a, g_x_a, M_a, Mr_a, t_a = _score(ai_records, ai_idx_all)
    print(f"  [corpus-feat] scoring {len(hu_idx_all)} Human samples ...", flush=True)
    g_ai_h, g_hu_h, g_x_h, M_h, Mr_h, t_h = _score(hu_records, hu_idx_all)
    ms_per_text = 1000.0 * (t_a + t_h) / (len(ai_idx_all) + len(hu_idx_all))

    payload = {
        "M_ai": M_a, "M_human": M_h,
        "M_raw_ai": Mr_a, "M_raw_human": Mr_h,
        "g_ai_given_ai": g_ai_a, "g_ai_given_human": g_hu_a,
        "g_human_given_ai": g_ai_h, "g_human_given_human": g_hu_h,
        "g_x_ai": g_x_a, "g_x_human": g_x_h,
        "ai_indices": np.array(ai_idx_all, dtype=np.int64),
        "human_indices": np.array(hu_idx_all, dtype=np.int64),
        "g_D_ai": np.array([g_D_ai]), "g_D_human": np.array([g_D_hu]),
        "ms_per_text": np.array([ms_per_text]),
        "train_build_s": np.array([train_build_s]),
    }
    np.savez_compressed(cache_file, **payload)
    print(f"  [corpus-feat] saved -> {cache_file}  ({ms_per_text:.2f} ms/text)")
    return payload


def build_feature_matrix(features: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_ai, X_human, ai_indices, human_indices) where each X is (n, 5).
    Columns: [M, M_raw, g_x_given_ai, g_x_given_human, g_x]."""
    X_ai = np.column_stack([
        features["M_ai"], features["M_raw_ai"],
        features["g_ai_given_ai"], features["g_ai_given_human"], features["g_x_ai"],
    ])
    X_hu = np.column_stack([
        features["M_human"], features["M_raw_human"],
        features["g_human_given_ai"], features["g_human_given_human"], features["g_x_human"],
    ])
    return X_ai, X_hu, features["ai_indices"], features["human_indices"]


def build_xy_for_split(features: Dict, split: Dict):
    X_ai, X_hu, ai_idx, hu_idx = build_feature_matrix(features)
    ai_list = ai_idx.tolist(); hu_list = hu_idx.tolist()
    tr_ai = [ai_list.index(t) for t in split["train_idx_ai"]]
    te_ai = [ai_list.index(t) for t in split["test_idx_ai"]]
    tr_hu = [hu_list.index(t) for t in split["train_idx_human"]]
    te_hu = [hu_list.index(t) for t in split["test_idx_human"]]
    X_tr = np.vstack([X_ai[tr_ai], X_hu[tr_hu]])
    X_te = np.vstack([X_ai[te_ai], X_hu[te_hu]])
    y_tr = np.array([1]*len(tr_ai) + [0]*len(tr_hu))
    y_te = np.array([1]*len(te_ai) + [0]*len(te_hu))
    return X_tr, y_tr, X_te, y_te, float(features["ms_per_text"][0])
