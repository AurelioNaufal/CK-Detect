"""Method 1: Rewrite-Conditioned Kolmogorov Complexity (with Raidar surface features).

Train a logistic regression on the 24-d aggregated feature vector, report
F1/AUC/Acc and ms/text.
"""
from __future__ import annotations

import argparse

from common import (ALL_DOMAINS, REWRITERS, fit_logreg, has_rewriter, make_split,
                    save_result, set_seed)
from features_rewrite import build_xy_for_split, compute_features


def run(domain: str, rewriter: str, overwrite_features: bool = False):
    if not has_rewriter(domain, rewriter):
        print(f"[skip] {domain}/{rewriter}: no rewrite file"); return None
    set_seed()
    split = make_split(domain, rewriter)
    feats = compute_features(domain, rewriter, overwrite=overwrite_features)
    X_tr, y_tr, X_te, y_te, ms = build_xy_for_split(feats, split)
    res = fit_logreg(X_tr, y_tr, X_te, y_te, ms_per_text=ms,
                     cache_key=("rewrite_conditioned", domain, rewriter))
    payload = {"method": "rewrite_conditioned", "domain": domain,
               "rewriter": rewriter,
               "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
               "feature_dim": int(X_tr.shape[1]), **res}
    save_result("rewrite_conditioned", domain, rewriter, payload)
    print(f"[rewrite] {domain}/{rewriter}: "
          f"F1={res['f1']:.4f}  AUC={res['auc']:.4f}  "
          f"Acc={res['accuracy']:.4f}  ms/text={ms:.2f}")
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
