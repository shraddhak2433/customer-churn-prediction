"""Customer Churn Analysis & Prediction
Dataset: Telco Customer Churn (IBM sample, on Kaggle). Save it as telco_churn.csv next to this file.
Run: python churn_analysis.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load and clean
df = pd.read_csv("telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks become NaN
print("Rows:", len(df), "| Missing TotalCharges:", df["TotalCharges"].isna().sum())
df = df.dropna(subset=["TotalCharges"]).drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print("Churn rate: {:.1%}".format(df["Churn"].mean()))

# 2. Exploratory charts
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df.groupby("Contract")["Churn"].mean().mul(100).plot.bar(ax=ax[0], color="#1d5f8a")
ax[0].set_title("Churn % by contract type"); ax[0].set_ylabel("%")
df.boxplot(column="tenure", by="Churn", ax=ax[1])
ax[1].set_title("Tenure (months) by churn"); ax[1].set_xlabel("Churn (0 = stayed, 1 = left)")
df.boxplot(column="MonthlyCharges", by="Churn", ax=ax[2])
ax[2].set_title("Monthly charges by churn"); ax[2].set_xlabel("Churn (0 = stayed, 1 = left)")
plt.suptitle(""); plt.tight_layout(); plt.savefig("churn_eda.png", dpi=150)

# 3. Model
X, y = df.drop(columns="Churn"), df["Churn"]
num = X.select_dtypes("number").columns
cat = X.select_dtypes(exclude="number").columns
prep = ColumnTransformer([("num", StandardScaler(), num),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {"Logistic Regression": LogisticRegression(max_iter=1000),
          "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)}
lines = []
for name, m in models.items():
    pipe = Pipeline([("prep", prep), ("model", m)]).fit(X_tr, y_tr)
    pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    lines.append(f"== {name} ==\nROC-AUC: {roc_auc_score(y_te, proba):.3f}\n"
                 f"{classification_report(y_te, pred)}")
    if name == "Random Forest":
        names = pipe["prep"].get_feature_names_out()
        imp = pd.Series(pipe["model"].feature_importances_, index=names).nlargest(10)
        plt.close("all")  # start a fresh figure so it does not draw over the EDA charts
        plt.figure(figsize=(8, 5))
        imp.sort_values().plot.barh(color="#1d5f8a")
        plt.title("Top 10 factors linked to churn"); plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)

report = "\n".join(lines)
print(report)
open("results.txt", "w").write(report)
