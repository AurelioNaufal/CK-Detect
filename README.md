# CK-Detect: AI-Generated Text Detection via Conditional Kolmogorov Complexity

This repository contains the code and datasets for **CK-Detect**, a compression-based framework for detecting AI-generated text. CK-Detect combines two Kolmogorov-complexity signals — rewrite-conditioned and corpus-conditioned — into a unified logistic regression classifier.

## Overview

CK-Detect uses two complementary signals:

1. **Rewrite-Conditioned**: Rewrites a suspect text with an LLM and measures the compression distance between original and rewrite. AI-generated text compresses closer to its rewrite (same statistical mode); human text diverges.

2. **Corpus-Conditioned (M-score)**: Compresses the suspect text against class-specific reference corpora (D_AI, D_Human) built from training data. The signed normalized difference reveals which distribution the text belongs to.

Both signals are combined into a single logistic regression (CK-Detect). Each signal can also be run standalone.

## Repository Structure

```
ck-detect/
├── code/
│   ├── common.py                    # Shared infrastructure (paths, splits, metrics)
│   ├── features_rewrite.py          # Feature extraction for rewrite-conditioned
│   ├── features_corpus.py           # Feature extraction for corpus-conditioned
│   │
│   ├── run_ck_detect.py             # CK-Detect (combined signal)
│   ├── run_rewrite_conditioned.py   # Rewrite-Conditioned only
│   └── run_corpus_conditioned.py    # Corpus-Conditioned only
│
└── Datasets/
    ├── Yelp/        # Yelp reviews (AI: GPT-3.5 generated)
    ├── Arxiv/       # ArXiv abstracts
    ├── Code/        # Code comments / docstrings
    ├── Essay/       # Student essays
    ├── CWriting/    # Creative writing
    └── News/        # News articles
```

## Requirements

```bash
pip install numpy scikit-learn fuzzywuzzy python-Levenshtein
```

No GPU required. `fuzzywuzzy` is used for one surface-similarity feature in the rewrite-conditioned branch; the code falls back to a plain-token overlap score if it isn't installed.

**Models used:**
- GPT-3.5-turbo or Qwen (via Ollama) — for rewriting text. Rewrite files for the 6 main domains are pre-generated and included in this repository, so a rewriter is only needed if you want to regenerate rewrites or add a new dataset.

## Dataset Format

Each dataset file is a JSON list of objects. Every object has an `input` field (the original text) plus one or more rewrite keys (one per prompt):

```json
[
  {
    "input": "Original text here...",
    "prompt_1": "Rewritten version using prompt 1...",
    "prompt_2": "Rewritten version using prompt 2..."
  },
  ...
]
```

`common.py` maps each domain to its AI and Human dataset files. Rewrites are pre-generated; to add a new rewriter, generate rewrite files in the same format and register them in `DATASET_FILES` in `common.py`.

## Running CK-Detect

All scripts are run from the `code/` directory. They automatically create a `cache/` folder for splits and features, and print/save an F1, AUC, and accuracy result for the requested domain.

```bash
cd code/

# CK-Detect (combined signal) — GPT rewriter, Yelp domain
python run_ck_detect.py --domain yelp --rewriter gpt

# Rewrite-Conditioned only
python run_rewrite_conditioned.py --domain yelp --rewriter gpt

# Corpus-Conditioned only
python run_corpus_conditioned.py --domain yelp --rewriter gpt
```

Available domains: `yelp`, `arxiv`, `code`, `essay`, `cwriting`, `news`
Available rewriters: `gpt`, `qwen`, `qwen08b`

To run across every domain, loop over them:

```bash
for d in yelp arxiv code essay cwriting news; do
  python run_ck_detect.py --domain "$d" --rewriter gpt
done
```

Each run writes its result to `results/<method>/<domain>_<rewriter>.json`.

Runtime is dominated by gzip compression and a logistic-regression fit — a few minutes per domain on a single CPU core.

## Citation

If you use this code, please cite:

```bibtex
@article{ckdetect2026,
  title   = {Conditional Kolmogorov Complexity for AI-Generated Text Detection},
  author  = {Aurelio Naufal Effendy, Hsing-Kuo Pao, Ghaluh Indah Permata Sari},
  year    = {2026},
}
```
