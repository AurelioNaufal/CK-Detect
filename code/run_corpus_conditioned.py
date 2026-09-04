"""Method 2: Corpus-Conditioned. Reports the zero-threshold classifier only:
    predict AI iff M(x) < 0   (paper's primary classifier; threshold-free).
AUC is computed from the raw score (-M, so higher = more AI-like).
"""
from __future__ import annotations

import argparse
import numpy as np

from common import (ALL_DOMAINS, REWRITERS, evaluate, has_rewriter, make_split,
                    save_predictions, save_result, set_seed)
from features_corpus import build_xy_for_split, compute_features


def run(domain: str, rewriter: str, overwrite_features: bool = False):
    if not has_rewriter(domain, rewriter):
        print(f"[skip] {domain}/{rewriter}: no rewrite file"); return None
    set_seed()
    split = make_split(domain, rewriter)
    feats = compute_features(domain, rewriter, overwrite=overwrite_features)
    _, _, X_te, y_te, ms = build_xy_for_split(feats, split)

    # Zero-threshold on M (col 0). No training needed.
    M_te = X_te[:, 0]
    y_pred = (M_te < 0).astype(int)
    y_score = -M_te  # higher = more AI
    save_predictions("corpus_conditioned", domain, rewriter, y_te, y_score, y_pred)
    res = evaluate(y_te, y_score, y_pred, ms_per_text=ms)

    # "Training" for corpus-conditioned = one-time gzip of D_AI/D_H (no classifier fit).
    train_time_s = float(feats.get("train_build_s", np.array([0.0]))[0])
    res["train_time_s"] = train_time_s
    payload = {
        "method": "corpus_conditioned", "domain": domain, "rewriter": rewriter,
        "n_test": int(len(y_te)),
        "classifier": "zero_threshold_M<0",
        **res,
    }
    save_result("corpus_conditioned", domain, rewriter, payload)
    print(f"[corpus] {domain}/{rewriter}: "
          f"F1={res['f1']:.4f}  AUC={res['auc']:.4f}  Acc={res['accuracy']:.4f}  "
          f"(M<0)  ms/text={ms:.2f}")
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
