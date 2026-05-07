"""
Credit Card Approval - Delinquency Prediction
Logistic Regression model on Kaggle Credit Card Approval dataset.

Dataset: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data
Place application_record.csv and credit_record.csv in the same folder as this script.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ── 1. Load data ──────────────────────────────────────────────────────────────

app = pd.read_csv("application_record.csv")
cr  = pd.read_csv("credit_record.csv")

print(f"Application records : {app.shape}")
print(f"Credit records      : {cr.shape}")

# ── 2. Build binary target ────────────────────────────────────────────────────
# BAD = ever 60+ days past due (STATUS in 2, 3, 4, 5)

cr["BAD_MONTH"] = cr["STATUS"].isin(["2", "3", "4", "5"]).astype(int)
target = cr.groupby("ID")["BAD_MONTH"].max().reset_index()
target.columns = ["ID", "bad_client"]

print(f"\nTarget distribution:\n{target['bad_client'].value_counts()}")

# ── 3. Merge & clean ──────────────────────────────────────────────────────────

app = app.drop_duplicates(subset="ID", keep="first")
df  = app.merge(target, on="ID", how="inner")

df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

print(f"\nMerged dataset shape: {df.shape}")
print(f"Default rate: {df['bad_client'].mean()*100:.2f}%")

# ── 4. Feature engineering ────────────────────────────────────────────────────

df["AGE_YEARS"]     = (-df["DAYS_BIRTH"] / 365).round(1)
df["IS_EMPLOYED"]   = (df["DAYS_EMPLOYED"] < 0).astype(int)
df["YEARS_EMPLOYED"]= np.where(df["DAYS_EMPLOYED"] < 0, -df["DAYS_EMPLOYED"] / 365, 0).round(1)
df["INCOME_K"]      = (df["AMT_INCOME_TOTAL"] / 1000).round(1)
df["GENDER_M"]      = (df["CODE_GENDER"] == "M").astype(int)
df["OWN_CAR"]       = (df["FLAG_OWN_CAR"] == "Y").astype(int)
df["OWN_REALTY"]    = (df["FLAG_OWN_REALTY"] == "Y").astype(int)
df["HIGH_EDU"]      = df["NAME_EDUCATION_TYPE"].isin(["Higher education", "Academic degree"]).astype(int)
df["MARRIED"]       = df["NAME_FAMILY_STATUS"].isin(["Married", "Civil marriage"]).astype(int)

features = [
    "AGE_YEARS", "YEARS_EMPLOYED", "INCOME_K", "CNT_FAM_MEMBERS",
    "GENDER_M", "OWN_CAR", "OWN_REALTY", "HIGH_EDU", "MARRIED", "IS_EMPLOYED",
]

X = df[features]
y = df["bad_client"]

# ── 5. Train / Validation / Test split (60 / 20 / 20, stratified) ─────────────

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print(f"\nTrain : {X_train.shape}  bad rate: {y_train.mean()*100:.2f}%")
print(f"Val   : {X_val.shape}    bad rate: {y_val.mean()*100:.2f}%")
print(f"Test  : {X_test.shape}   bad rate: {y_test.mean()*100:.2f}%")

# ── 6. Logistic regression (statsmodels) ──────────────────────────────────────

X_train_sm = sm.add_constant(X_train)
X_val_sm   = sm.add_constant(X_val)
X_test_sm  = sm.add_constant(X_test)

model = sm.Logit(y_train, X_train_sm).fit(disp=0)
print("\n" + str(model.summary()))
print(f"\nPseudo R²  : {model.prsquared:.4f}")
print(f"AIC        : {model.aic:.2f}")
print(f"BIC        : {model.bic:.2f}")
print(f"LLR p-value: {model.llr_pvalue:.4e}")

# Coefficients & odds ratios
coef_table = pd.DataFrame({
    "Coef"     : model.params.round(4),
    "Std.Err"  : model.bse.round(4),
    "z"        : model.tvalues.round(3),
    "P>|z|"   : model.pvalues.round(4),
    "OddsRatio": np.exp(model.params).round(3),
})
print(f"\n{coef_table}")

# ── 7. Predicted probabilities ────────────────────────────────────────────────

prob_train = model.predict(X_train_sm)
prob_val   = model.predict(X_val_sm)
prob_test  = model.predict(X_test_sm)

# ── 8. ROC curves & AUC ───────────────────────────────────────────────────────

fpr_tr, tpr_tr, _       = roc_curve(y_train, prob_train)
fpr_va, tpr_va, thr_va  = roc_curve(y_val,   prob_val)
fpr_te, tpr_te, _       = roc_curve(y_test,  prob_test)

auc_tr = auc(fpr_tr, tpr_tr)
auc_va = auc(fpr_va, tpr_va)
auc_te = auc(fpr_te, tpr_te)

print(f"\nAUC  Train: {auc_tr:.4f}")
print(f"AUC  Val  : {auc_va:.4f}")
print(f"AUC  Test : {auc_te:.4f}")

# ── 9. Youden-optimal threshold (on validation set) ───────────────────────────

j_scores = tpr_va - fpr_va
opt_thr  = thr_va[np.argmax(j_scores)]
print(f"\nYouden-optimal threshold (validation): {opt_thr:.4f}")

# ── 10. Evaluation at chosen threshold ───────────────────────────────────────

def metrics_at(y_true, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    acc  = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    cm   = np.array([[tn, fp], [fn, tp]])
    return cm, acc, prec, rec, spec

cm_default, acc_d, pr_d, rec_d, sp_d = metrics_at(y_test, prob_test, 0.5)
cm_tuned,   acc_t, pr_t, rec_t, sp_t = metrics_at(y_test, prob_test, opt_thr)

print("\n=== Threshold = 0.50 (default) ===")
print(f"Accuracy: {acc_d:.4f}  Precision: {pr_d:.4f}  Recall: {rec_d:.4f}  Specificity: {sp_d:.4f}")
print(cm_default)

print(f"\n=== Threshold = {opt_thr:.3f} (Youden optimum) ===")
print(f"Accuracy: {acc_t:.4f}  Precision: {pr_t:.4f}  Recall: {rec_t:.4f}  Specificity: {sp_t:.4f}")
print(cm_tuned)

# ── 11. Figures ───────────────────────────────────────────────────────────────

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_tr, tpr_tr, color="steelblue",  lw=2, label=f"Train (AUC = {auc_tr:.3f})")
ax.plot(fpr_va, tpr_va, color="darkorange", lw=2, label=f"Validation (AUC = {auc_va:.3f})")
ax.plot(fpr_te, tpr_te, color="firebrick",  lw=2, label=f"Test (AUC = {auc_te:.3f})")
ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random (AUC = 0.5)")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC Curve"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("fig_roc.png", dpi=150); plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, cm, title in [
    (axes[0], cm_default, "Threshold = 0.50"),
    (axes[1], cm_tuned,   f"Threshold = {opt_thr:.3f} (Youden)"),
]:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=150); plt.show()

# Coefficients
params = model.params.drop("const")
colors = ["firebrick" if p < 0.05 else "lightgray" for p in model.pvalues.drop("const")]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(params.index, params.values, color=colors)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Coefficient (log-odds)")
ax.set_title("Logistic regression coefficients  (red = p < 0.05)")
plt.tight_layout(); plt.savefig("fig_coefs.png", dpi=150); plt.show()

print("\nDone. Figures saved to current directory.")
