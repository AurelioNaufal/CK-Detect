"""CK-Detect: rewrite-conditioned (24-d) + corpus-conditioned M-score (1-d)
fed to a single logistic regression. Total 25-d.
ms/text = rewrite_ms + corpus_ms.
"""
from __future__ import annotations

import argparse
import numpy as np

from common import (ALL_DOMAINS, REWRITERS, fit_logreg, has_rewriter, make_split,
                    save_result, set_seed)
from features_rewrite import compute_features as compute_rewrite_features
from features_corpus  import compute_features as compute_corpus_features


def _align(features_rewrite, features_corpus, split):
    """Return (X_ai, X_human) combined matrices aligned to the (rewrite) indices."""
    re_ai_idx = features_rewrite["ai_indices"].tolist()
    re_hu_idx = features_rewrite["human_indices"].tolist()
    co_ai_idx = features_corpus["ai_indices"].tolist()
    co_hu_idx = features_corpus["human_indices"].tolist()

    # Build 1-D M-score column aligned to rewrite ordering.
    def align_M(M, co_idx, target):
        co_pos = {g: i for i, g in enumerate(co_idx)}
        return np.array([M[co_pos[g]] for g in target], dtype=np.float64).reshape(-1, 1)

    M_ai_aligned = align_M(features_corpus["M_ai"],    co_ai_idx, re_ai_idx)
    M_hu_aligned = align_M(features_corpus["M_human"], co_hu_idx, re_hu_idx)

    X_ai = np.hstack([features_rewrite["X_ai"],    M_ai_aligned])
    X_hu = np.hstack([features_rewrite["X_human"], M_hu_aligned])
    return X_ai, X_hu, re_ai_idx, re_hu_idx


def run(domain: str, rewriter: str, overwrite_features: bool = False):
    if not has_rewriter(domain, rewriter):
        print(f"[skip] {domain}/{rewriter}: no rewrite file"); return None
    set_seed()
    split = make_split(domain, rewriter)
    f_re = compute_rewrite_features(domain, rewriter, overwrite=overwrite_features)
    f_co = compute_corpus_features(domain, rewriter, overwrite=overwrite_features)
    X_ai, X_hu, ai_idx, hu_idx = _align(f_re, f_co, split)

    tr_ai = [ai_idx.index(t) for t in split["train_idx_ai"]]
    te_ai = [ai_idx.index(t) for t in split["test_idx_ai"]]
    tr_hu = [hu_idx.index(t) for t in split["train_idx_human"]]
    te_hu = [hu_idx.index(t) for t in split["test_idx_human"]]
    X_tr = np.vstack([X_ai[tr_ai], X_hu[tr_hu]])
    X_te = np.vstack([X_ai[te_ai], X_hu[te_hu]])
    y_tr = np.array([1]*len(tr_ai) + [0]*len(tr_hu))
    y_te = np.array([1]*len(te_ai) + [0]*len(te_hu))

    ms = float(f_re["ms_per_text"][0]) + float(f_co["ms_per_text"][0])
    res = fit_logreg(X_tr, y_tr, X_te, y_te, ms_per_text=ms,
                     cache_key=("ck_detect", domain, rewriter))
    # CK-Detect inherits the corpus-build cost from corpus-conditioned: D_AI / D_H
    # must be compressed at training time so the rewrite features can be combined
    # with the M-score at inference. Add it to the LR-fit time.
    corpus_build_s = float(f_co.get("train_build_s", np.array([0.0]))[0])
    res["train_time_s"] = float(res["train_time_s"]) + corpus_build_s
    payload = {"method": "ck_detect", "domain": domain, "rewriter": rewriter,
               "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
               "feature_dim": int(X_tr.shape[1]), **res}
    save_result("ck_detect", domain, rewriter, payload)
    print(f"[CK-Detect] {domain}/{rewriter}: "
          f"F1={res['f1']:.4f}  AUC={res['auc']:.4f}  Acc={res['accuracy']:.4f}  "
          f"ms/text={ms:.2f}  (feat_dim={X_tr.shape[1]})")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=ALL_DOMAINS)
    p.add_argument("--rewriter", choices=REWRITERS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--overwrite-features", action="store_true")
    args = p.parse_args()

    if args.all:
        for d in ALL_DOMAINS:
            for r in REWRITERS:
                run(d, r, args.overwrite_features)
    else:
        run(args.domain, args.rewriter, args.overwrite_features)


if __name__ == "__main__":
    main()
