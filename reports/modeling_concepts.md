# Modeling Concepts & Results — Cancellation Classification

## Results (test set: 39,390 bookings held out from an 80,000-row stratified train)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|-------|----------|-----------|--------|-----|---------|-----------|
| **XGBoost** | **0.882** | 0.861 | **0.813** | **0.836** | **0.954** | 1.2s |
| Random Forest | 0.879 | 0.891 | 0.767 | 0.825 | 0.953 | 2.0s |
| MLP (deep learning) | 0.863 | 0.839 | 0.781 | 0.809 | 0.939 | 18.7s |
| Decision Tree | 0.848 | 0.838 | 0.732 | 0.782 | 0.927 | 0.9s |
| Logistic Regression | 0.806 | 0.798 | 0.638 | 0.709 | 0.874 | 14.3s |

Full per-model classification reports: `classification_reports.md`. Confusion
matrices: `confusion_matrix_*.png`. Raw table: `classification_model_comparison.csv`.

**Best model: XGBoost** — highest ROC-AUC (0.954) and F1 (0.836), and one of the
fastest to train. Random Forest is essentially tied; XGBoost edges it on recall and
speed.

## Linear vs non-linear data

- **Linear data** means the classes can be separated well by a straight line / flat
  hyperplane — a weighted sum of features (`w·x + b`) is enough.
- **Non-linear data** means the decision boundary curves and bends; class membership
  depends on **interactions and thresholds** between features, not a single weighted
  sum.
- **Logistic Regression is a linear model**: it fits one hyperplane, so it can only
  capture "more of feature X → more likely to cancel" effects. It is our baseline.
- **Decision Tree, Random Forest, XGBoost and MLP are non-linear**: trees split the
  space into rectangular regions and stack thresholds (capturing interactions for
  free); the MLP composes layers of non-linear (ReLU) activations to bend the
  boundary.

**What the results say:** every non-linear model beats Logistic Regression by a large
margin (ROC-AUC 0.93–0.95 vs 0.87; F1 0.78–0.84 vs 0.71). Combined with the EDA
finding that no numeric feature is strongly linearly correlated with the target, this
is strong evidence that **cancellation behaviour is non-linear** — driven by
interactions and thresholds (e.g. *Non-Refund deposit* almost always cancels;
*long lead time AND online segment AND no special requests* compound risk).

## Bias and variance

- **Logistic Regression — high bias, low variance.** It underfits this non-linear
  problem: too simple to bend around the boundary, so train and test scores are both
  modest but stable. (Its slow lbfgs convergence here is a side effect of *not*
  scaling features — a project requirement.)
- **Decision Tree — low bias, high variance.** A single deep tree fits training data
  closely but is sensitive to it; it can overfit (here we cap `max_depth=12` to limit
  that). Lowest of the non-linear models.
- **Random Forest — reduces variance.** Averaging 200 de-correlated trees (bagging)
  cancels out individual trees' overfitting, giving a big jump over a single tree.
- **XGBoost — reduces bias and controls variance.** Boosting builds trees
  sequentially, each correcting the previous errors (lowering bias), with
  learning-rate shrinkage, subsampling and regularization holding variance down.
  Best overall here.
- **MLP — learns complex patterns, can overfit.** The neural net captures non-linear
  structure well but needs scaled inputs (scaling is applied *only* inside its own
  pipeline), is slower to train, and is prone to overfit without `early_stopping`
  (which we enable).

**Underfitting** = model too simple, high bias, poor on both train and test (Logistic
Regression here). **Overfitting** = model too complex, high variance, great on train
but worse on test (an unpruned Decision Tree). **Generalization** = the goal in
between: strong, *consistent* performance on unseen data — which the ensembles
(Random Forest, XGBoost) achieve best.

## Evaluation techniques (and why accuracy alone is not enough)

- **Accuracy** — overall fraction correct. **Misleading under class imbalance**: 37%
  of bookings cancel, so a model that *never* predicts "cancel" already scores ~63%
  while being useless. Accuracy hides which class is wrong.
- **Precision** — of the bookings we flag as "will cancel", how many actually do.
  High precision = few false alarms.
- **Recall** — of the bookings that actually cancel, how many we catch. **Most
  important here**: a missed cancellation (false negative) is the costly error for
  revenue/overbooking decisions.
- **F1-score** — harmonic mean of precision and recall; the headline single number
  when both errors matter and classes are imbalanced.
- **ROC-AUC** — ranking quality of the predicted **probabilities** across all
  thresholds; threshold-independent and robust to imbalance. **Our primary model
  selection metric** because the project outputs a *probability* of cancellation.
- **Confusion Matrix** — the raw TP/FP/FN/TN counts behind every metric above
  (`confusion_matrix_*.png`).
- **Classification Report** — precision/recall/F1 per class in one view
  (`classification_reports.md`).

For this project we prioritise **recall, F1, ROC-AUC and well-calibrated probability
output** over raw accuracy.
