# Credit Card Approval — Delinquency Prediction

Logistic regression model predicting whether a credit card applicant will become a bad client (60+ days past due), built for an Applied Business Statistics course.

## Dataset

[Credit Card Approval Prediction — Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data)

Two files are required (place them in the same folder as the script):

| File | Contents |
|---|---|
| `application_record.csv` | One row per applicant — demographics, income, family info |
| `credit_record.csv` | Monthly repayment status per applicant |

## Model overview

| Step | Detail |
|---|---|
| Target | Binary: bad client = ever 60+ days past due (STATUS 2–5) |
| Features | Age, employment, income, family size, gender, car/realty ownership, education, marital status |
| Algorithm | Logistic regression (`statsmodels.Logit`) |
| Split | Stratified 60 / 20 / 20 (train / validation / test) |
| Threshold | Youden's J optimum on validation set (not default 0.5) |

## Results

- **AUC ≈ 0.57** on the test set — realistic given only demographic/financial features and no behavioural credit history
- Default threshold (0.5) is unusable at ~1–2% base rate: flags almost no bad clients
- Youden-optimal threshold recovers ~50% of true bad clients at the cost of more false positives

## Requirements

```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

## Usage

```bash
python credit_approval_model.py
```

The script prints the full model summary, AUC scores, and confusion matrices, and saves three figures to the current directory: `fig_roc.png`, `fig_confusion.png`, `fig_coefs.png`.

## Report

See `Report.pdf` for the full write-up including methodology, visualisations, and interpretation.
