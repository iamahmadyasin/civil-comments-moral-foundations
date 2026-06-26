# Do Moral Foundations Predict Online Toxicity?
### A Bayesian Hurdle Model Analysis of 10,000 Civil Comments

![Master Summary](outputs/figures/04_master_summary.png)

---

## Research Question

> Do comments exhibiting moral-emotional language patterns, as identified by Moral Foundations Theory, receive systematically higher toxicity ratings, and does this effect vary across moral dimensions (care, fairness, loyalty, authority, purity)?

---

## Key Findings

1. **Purity and Authority** are the strongest predictors of a comment crossing the toxicity threshold. Disgust and defiance-of-authority framing reliably signal toxic content.

2. **Authority dominates toxicity intensity.** Once a comment is toxic, defiance-of-authority language predicts how extreme it becomes — more than any other moral foundation.

3. **Loyalty shows a consistent negative effect** in both model components, the only foundation to do so. Comments using loyalty or betrayal framing are less likely to be toxic and less intense when flagged.

4. **Care and Fairness predict threshold-crossing but not intensity**, suggesting cruelty and unfairness framing marks a comment as harmful without determining its severity.

5. **A standard regression would have obscured findings 3 and 4** by averaging across both processes. The two-part hurdle model was necessary to reveal the distinct roles of each moral foundation.


---

## Results Summary

| Foundation | Correlation (r) | Hurdle Effect | Intensity Effect |
|:-----------|:--------------:|:-------------:|:----------------:|
| Purity     | 0.304          | +0.33         | +0.13            |
| Authority  | 0.260          | +0.33         | +0.19            |
| Care       | 0.258          | +0.19         | ~0               |
| Fairness   | 0.236          | +0.14         | ~0               |
| Loyalty    | 0.167          | **−0.24**     | **−0.05**        |

> Hurdle and intensity effects are posterior means on standardized predictors.
> ~0 indicates the 94% HDI spans zero (uncertain direction).

---

## Dataset

**Google Civil Comments:** 1.8 million real online comments with human-annotated
toxicity scores from 0.0 to 1.0, across six dimensions.

| Property | Detail |
|:---------|:-------|
| Source | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) on Hugging Face |
| Sample used | 10,000 comments from the training split |
| License | CC0 1.0 (public domain) |
| Toxicity dimensions | toxicity, severe_toxicity, obscene, threat, insult, identity_attack |

**Key characteristic:** 74.8% of comments score exactly zero on toxicity which is 
a severe zero-inflation that makes standard regression inappropriate and 
motivates the hurdle model approach.

---

## Methods

### Step 1: NLP Feature Engineering

Each comment was scored on five moral dimensions using zero-shot classification
(`facebook/bart-large-mnli`). Moral Foundations Theory (Haidt & Graham, 2007)
posits that human moral reasoning organizes around five innate foundations.

| Foundation | Vice Label Used as Predictor |
|:-----------|:-----------------------------|
| Care | cruel, harmful, causing pain or suffering |
| Fairness | unfair, cheating, discrimination |
| Loyalty | betrayal, disloyalty, treason |
| Authority | disrespect, defiance, subversion of authority |
| Purity | disgusting, degrading, depraved, morally corrupt |

Each comment receives a raw vice score per foundation (0.0–1.0), used as the
predictor in the model.

### Step 2: Bayesian Hurdle Model

A two-part model addressing the zero-inflation in the outcome variable.

**Part 1: The Hurdle (Logistic Regression)**

Does moral-vice language predict whether a comment crosses the toxicity
threshold at all? Binary outcome: `toxicity > 0`.

**Part 2: The Intensity (Beta Regression)**

Among toxic comments only: does moral-vice language predict how extreme the
toxicity score is? Continuous outcome bounded (0, 1).

**Inference details:**

- Priors: `Normal(0, 1)` on all standardized predictors (weakly informative)
- Sampler: NUTS, 4 chains × 1,000 draws, 1,000 tuning steps
- Convergence: R-hat < 1.01 across all parameters
- Software: PyMC 5, ArviZ

---

## Project Structure

```
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Data loading and EDA
│   ├── 02_moral_foundations.ipynb       # NLP scoring pipeline (run on Colab)
│   ├── 03_bayesian_model.ipynb          # Hurdle model (run on Colab)
│   └── 04_visualization.ipynb          # Summary figures
├── src/
│   └── config.py                        # Shared paths and settings
├── data/
│   ├── 01_score_summary.csv             # Descriptive statistics from EDA
│   └── 02_civil_comments_with_mf_scores.csv  # Enriched dataset with MF scores
├── outputs/
│   └── figures/                         # All saved charts
├── requirements.txt
└── README.md
```

> **Note:** Posterior trace files (`.nc`) are not included due to file size.
> They are regenerated automatically by running `03_bayesian_model.ipynb`.

---

## Reproducing This Project

### Prerequisites

- Python 3.10+
- A free [Hugging Face account](https://huggingface.co) and read token
- A free [Google account](https://colab.research.google.com) for Colab (GPU required for notebooks 02–03)

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/iamahmadyasin/civil-comments-moral-foundations.git
cd civil-comments-moral-foundations
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Hugging Face token**

Open `src/config.py` and replace the placeholder:
```python
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
```

### Running the Notebooks

| Notebook | Where to run | Runtime |
|:---------|:------------|:--------|
| `01_data_exploration.ipynb` | Local (VS Code / Jupyter) | ~5 min |
| `02_moral_foundations.ipynb` | Google Colab — T4 GPU | ~60 min |
| `03_bayesian_model.ipynb` | Google Colab — T4 GPU | ~20 min |
| `04_visualization.ipynb` | Google Colab or local | ~1 min |

**To enable the GPU in Colab:**
Runtime → Change runtime type → T4 GPU → Save

---

## Tools and Libraries

| Purpose | Library / Tool |
|:--------|:--------------|
| Data streaming | `datasets` (Hugging Face) |
| NLP classification | `transformers`, `torch` |
| Bayesian modeling | `pymc`, `arviz` |
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Compute | Google Colab (T4 GPU, free tier) |

---

## References

- Haidt, J., & Graham, J. (2007). When morality opposes justice: Conservatives
  have moral intuitions that liberals may not recognize. *Social Justice Research, 20*(1), 98–116.

- Borkan, D. et al. (2019). Nuanced metrics for measuring unintended bias with
  real data for text classification. *WWW '19 Companion.*
  [arxiv.org/abs/1903.04561](https://arxiv.org/abs/1903.04561)

- Lewis, M. et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training
  for Natural Language Generation, Translation, and Comprehension.
  [arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
