# C2DB-T: Correlation & Calibration-aware Distribution-Balanced Transformer

**Breaking the Mutual Exclusivity and Imbalance Constraints in Multi-Label Text Classification**

📄 *Submitted for journal review* · Zahid Mehmood, Muhammad Ali Irtaza, **Faiza Fatima** — University of Engineering and Technology, Taxila

[![Micro-F1](https://img.shields.io/badge/EUR--Lex%20Micro--F1-0.7826-brightgreen)]()
[![Macro-F1 Gain](https://img.shields.io/badge/Macro--F1%20Gain-%2B5.72%25-blue)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-RoBERTa%20backbone-orange)]()

---

## ✨ Overview

Multi-label text classification (MLTC) breaks down in the real world for three stubborn reasons: **long-tail class imbalance**, **hidden dependencies between labels**, and a **one-size-fits-all decision threshold** that quietly punishes rare classes. Most existing fixes solve one of these problems while ignoring the other two — and usually at the cost of heavy architectural complexity.

**C2DB-T** unifies all three fixes into a single, lightweight, differentiable pipeline built on top of a frozen RoBERTa / TF-IDF feature extractor:

1. **Distribution-Balanced (DB) Loss** *(with a weight-clipping fix)* — reweights the loss to stop frequent labels from drowning out rare ones.
2. **Graph-based Label Correlation Module (LCM)** — an unparameterized, row-normalized label co-occurrence graph that propagates signal between related labels (e.g. *"clinical trial" → "regulatory approval"*).
3. **Adaptive Per-Label Threshold + Post-hoc Temperature Scaling** — replaces the naive global 0.5 cutoff with a learned, calibrated decision boundary per label.

The result is a framework that is **computationally tractable** (no end-to-end transformer fine-tuning required in its efficient mode) while still delivering **state-of-the-art gains on tail-label performance**, validated across four structurally different benchmark datasets.

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture_pipeline.png" alt="C2DB-T end-to-end pipeline" width="800"/>
</p>

<p align="center"><em>Figure — End-to-end C2DB-T pipeline: preprocessing → dual feature extraction (TF-IDF / RoBERTa) → Incremental PCA → backbone → DB Loss → Label Correlation Module → Adaptive Threshold → post-hoc calibration.</em></p>

**Pipeline stages:**

| Stage | What it does |
|---|---|
| **Preprocessing** | Unicode normalization, length filtering, vocabulary pruning, multi-label binarization — all statistics fit on train only to avoid leakage. |
| **Feature Extraction** | Sublinear TF-IDF for news-style corpora (RCV1, Events-Biotech) or frozen RoBERTa mean-pooled embeddings for context-heavy corpora (EUR-Lex, Enron). RoBERTa runs **once**, embeddings are cached — cutting training time from ~48h to ~2–4h. |
| **Incremental PCA** | Compresses high-dimensional features (up to 47,236-D TF-IDF / 768-D RoBERTa) down to a 512-D representation in memory-safe chunks, retaining ≥90% variance. |
| **Label Co-occurrence Graph** | A sparsified, row-normalized adjacency matrix built once from training labels — encodes domain knowledge as a fixed, unlearnable buffer. |
| **Backbone** | 512 → 1024 → 512 → L feed-forward network with LayerNorm + GELU + dropout (dataset-specific regularization strength). |
| **DB Loss** | Class-frequency-aware reweighted BCE with weight clipping to prevent gradient explosions on ultra-rare labels. |
| **Label Correlation Module** | Learnable scalar-gated correlation propagation between logits, seeded small so the backbone drives predictions early in training. |
| **Adaptive Threshold + Calibration** | Per-label learned offsets + post-hoc temperature scaling + cross-validated threshold tuning on a held-out split. |

Training uses a **two-stage schedule** (backbone warmup → joint fine-tuning) with a warmup-cosine LR schedule, AdamW, gradient clipping, and mixed-precision (FP16) training.

---

## 📊 Results

Evaluated on four benchmarks spanning legal, news, biomedical, and corporate-email domains — using identical metrics across the board: **Micro-F1, Macro-F1, Precision@K, and NDCG@K**.

| Dataset | Labels | Train N | Baseline Micro-F1 | **C2DB-T Micro-F1** | Baseline Macro-F1 | **C2DB-T Macro-F1** | Δ Micro-F1 |
|---|---|---|---|---|---|---|---|
| **EUR-Lex** (EU legal) | 100 | 55,000 | 0.7402 | **0.7826** | 0.6039 | **0.6835** | **+5.72%** |
| **RCV1** (newswire) | 103 | 23,149 | 0.8253 | 0.8252 | 0.5792 | **0.5832** | −0.01% |
| **Enron** (corporate email) | 20 | 31,716 | 0.8102 | **0.8151** | 0.7591 | **0.7631** | +0.60% |
| **Events-Biotech** (low-resource) | 29 | 2,660 | 0.7027 | 0.6770* | 0.5018 | **0.5236** | **+4.3%** (Macro-F1) |

*\*Full C2DB-T trades some Micro-F1 for a significant Macro-F1 gain on this dataset — a deliberate consequence of prioritizing rare-event retrieval, as discussed below.*

On EUR-Lex, C2DB-T also achieves **P@1 = 86.66%** and **NDCG@5 = 0.7922**, outperforming the Label Attention & Correlation Network (LACN) baseline. On RCV1's scale-heavy, distribution-shifted test set, LACN retains a slight Micro-F1 edge (0.8790 vs 0.8252) — an honest, reported trade-off rather than a cherry-picked win.

### Ablation studies

Each component (DB Loss → Label Correlation → Adaptive Threshold) is added progressively and evaluated independently on every dataset:

<p align="center">
  <img src="assets/ablation_eurlex.png" alt="C2DB-T ablation results on EUR-Lex" width="700"/>
</p>
<p align="center"><em>Progressive ablation on EUR-Lex — each module adds measurable gain over the BCE baseline.</em></p>

<p align="center">
  <img src="assets/ablation_enron.png" alt="C2DB-T ablation results on Enron" width="700"/>
</p>
<p align="center"><em>Progressive ablation on Enron — the framework's best absolute Macro-F1 result.</em></p>

**Key takeaways from the ablation:**
- **DB Loss** helps most on moderately sized, well-covered label spaces (Enron); it can trade Micro-F1 for Macro-F1 on very small corpora (Events-Biotech).
- **Label Correlation** delivers its single biggest jump (+3.23% Macro-F1) on the smallest label space (Events-Biotech, 29 labels), confirming that graph-based propagation is most valuable when supervision is scarce.
- **Adaptive Threshold Calibration** consistently and reliably boosts Macro-F1 across all four datasets by recalibrating per-label decision boundaries — the framework's most universal contribution.

---

## 🧠 Why this matters

Most MLTC literature chases a single axis of improvement — either imbalance, or correlation, or calibration — and the ones that combine all three tend to introduce heavy attention stacks or auxiliary networks. C2DB-T shows that a **frozen encoder + a lightweight, mostly non-parametric pipeline** (a fixed co-occurrence graph, a single learnable correlation scalar, per-label threshold offsets) can match or beat much heavier architectures, while staying interpretable and cheap enough to train on a single GPU in hours, not days.

---

## 📁 Repository Structure

```
├── data/                  # Dataset loading & preprocessing scripts
├── src/
│   ├── features/          # TF-IDF and RoBERTa embedding extraction
│   ├── pca/                # Incremental PCA dimensionality reduction
│   ├── graph/              # Label co-occurrence adjacency matrix construction
│   ├── model/              # Backbone, DB Loss, Label Correlation Module, Adaptive Threshold
│   ├── calibration/        # Temperature scaling + cross-validated threshold tuning
│   └── train.py             # Two-stage training loop
├── configs/                # Per-dataset hyperparameter configs (EUR-Lex, RCV1, Enron, Events-Biotech)
├── notebooks/               # Ablation study & results reproduction notebooks
├── assets/                 # Figures used in this README
└── README.md
```

> Update this section to match your actual repo layout before publishing.

---

## ⚙️ Getting Started

```bash
# Clone the repo
git clone https://github.com/Faiza2850/c2db-t.git
cd c2db-t

# Install dependencies
pip install -r requirements.txt

# Train on a benchmark (example: EUR-Lex)
python src/train.py --config configs/eurlex.yaml

# Run the full ablation sweep
python src/train.py --config configs/eurlex.yaml --ablation full
```

*(Add exact CLI flags / setup instructions once the training scripts are finalized in the repo.)*

---

## 📚 Datasets

| Dataset | Domain | Labels | Feature Type |
|---|---|---|---|
| [EUR-Lex](https://huggingface.co/datasets/jonathanli/eurlex) | EU legal documents | 100 | RoBERTa |
| [RCV1](https://scikit-learn.org/0.18/datasets/rcv1.html) | Reuters newswire | 103 | TF-IDF |
| [Enron](https://huggingface.co/datasets/corbt/enron-emails) | Corporate email | 20 | RoBERTa |
| [Events-Biotech](https://huggingface.co/datasets/knowledgator/events_classification_biotech) | Biotech news events | 29 | TF-IDF |

All datasets are publicly available; see the paper for exact split details.

---

## 🔬 Limitations & Future Work

- On RCV1 and Events-Biotech, C2DB-T doesn't beat the baseline on raw Micro-F1 — large train/test distribution shift and very small training corpora limit what post-hoc calibration alone can fix.
- The label co-occurrence graph is **static**, built once from training data. A dynamic or attention-based label graph is a natural next step.
- PCA compression costs ~4–5 Micro-F1 points versus full RoBERTa fine-tuning on EUR-Lex — parameter-efficient fine-tuning (e.g. LoRA) is a promising middle ground.

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{mehmood2026c2dbt,
  title   = {C2DB-T: Breaking the Mutual Exclusivity and Imbalance Constraints in Multi-Label Text Classification},
  author  = {Mehmood, Zahid and Irtaza, Muhammad Ali and Fatima, Faiza},
  journal = {Under Review},
  year    = {2026}
}
```

---

## 👤 Author

**Faiza Fatima**
BS Computer Science, University of Engineering and Technology, Taxila
📧 faizafatima213301@gmail.com · [GitHub](https://github.com/Faiza2850) · [LinkedIn](https://linkedin.com/in/faiza-fatima-824070251)

---

## 📄 License

Specify your license here (e.g. MIT, Apache 2.0) before publishing.
