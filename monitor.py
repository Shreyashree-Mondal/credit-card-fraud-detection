"""Transaction compliance monitor: CBLOF anomaly flags + rule-based checks + SHAP + audit log."""
import numpy as np
import pandas as pd
from pyod.models.cblof import CBLOF
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

df = pd.read_csv("creditcard_small.csv")
features = [c for c in df.columns if c not in ("Time", "Fraud")]
X = StandardScaler().fit_transform(df[features].values)

model = CBLOF(n_clusters=12, contamination=0.03, alpha=0.9, beta=3,
              use_weights=False, random_state=RANDOM_STATE)
model.fit(X)
df["cblof_score"] = model.decision_scores_
df["cblof_flag"] = model.labels_

flagged = int(df["cblof_flag"].sum())
caught = int(((df["cblof_flag"] == 1) & (df["Fraud"] == 1)).sum())
print("CBLOF flagged:", flagged, "of", len(df))
print("Fraud caught:", caught, "of", int(df["Fraud"].sum()))

# ---- Rule-based compliance checks (Amount only; f1-f28 are anonymized) ----
amount_p99 = df["Amount"].quantile(0.99)
df["rule_high_value"] = (df["Amount"] >= amount_p99).astype(int)
df["rule_round_amount"] = ((df["Amount"] >= 50) & (df["Amount"] % 10 == 0)).astype(int)
df["rule_zero_amount"] = (df["Amount"] == 0).astype(int)
rule_cols = ["rule_high_value", "rule_round_amount", "rule_zero_amount"]
df["rule_flag"] = (df[rule_cols].sum(axis=1) > 0).astype(int)
df["final_flag"] = ((df["cblof_flag"] == 1) | (df["rule_flag"] == 1)).astype(int)

print("High-value threshold (99th pct):", round(amount_p99, 2))
for c in rule_cols + ["rule_flag", "final_flag"]:
    n = int(df[c].sum())
    f = int(((df[c] == 1) & (df["Fraud"] == 1)).sum())
    print(f"{c}: {n} flagged, {f} fraud")

# ---- Review-queue tiers ----
df["priority"] = np.select(
    [(df["cblof_flag"] == 1) & (df["rule_flag"] == 1),
     (df["cblof_flag"] == 1),
     (df["rule_flag"] == 1)],
    ["HIGH", "MEDIUM", "LOW"], default="NONE")
print(pd.crosstab(df["priority"], df["Fraud"], margins=True))

# ---- SHAP explanations via a surrogate model ----
import shap
from sklearn.ensemble import RandomForestRegressor

surrogate = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=RANDOM_STATE)
surrogate.fit(X, df["cblof_score"].values)
print("Surrogate out-of-bag R2 vs CBLOF score:", round(surrogate.oob_score_, 3))

shap_values = shap.TreeExplainer(surrogate).shap_values(X)

def top_reasons(i, k=3):
    idx = np.argsort(-np.abs(shap_values[i]))[:k]
    return "; ".join(f"{features[j]} ({shap_values[i][j]:+.2f})" for j in idx)

df["shap_reasons"] = ""
for i in np.where(df["cblof_flag"] == 1)[0]:
    df.loc[i, "shap_reasons"] = top_reasons(i)

top = df[df["cblof_flag"] == 1].sort_values("cblof_score", ascending=False).head(5)
print(top[["cblof_score", "Fraud", "shap_reasons"]].to_string())

# ---- Audit log ----
from datetime import datetime, timezone
run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

df["rule_tags"] = df.apply(
    lambda r: ",".join(c.replace("rule_", "") for c in rule_cols if r[c] == 1), axis=1)
df["review_queue"] = np.where(df["cblof_flag"] == 1, "REVIEW",
                              np.where(df["rule_flag"] == 1, "SAMPLE", "NONE"))
df["txn_id"] = df.index

audit = df[df["review_queue"] != "NONE"].copy()
audit["queue_rank"] = (audit.groupby("review_queue")["cblof_score"]
                       .rank(ascending=False, method="first").astype(int))
audit["model"] = "CBLOF(n_clusters=12, contamination=0.03, alpha=0.9, beta=3, seed=42)"
audit["run_utc"] = run_ts

cols = ["run_utc", "txn_id", "Time", "Amount", "cblof_score", "cblof_flag",
        "rule_tags", "review_queue", "queue_rank", "shap_reasons", "model"]
audit.sort_values(["review_queue", "queue_rank"])[cols].to_csv("audit_log.csv", index=False)
print("audit_log.csv rows:", len(audit))
print(audit["review_queue"].value_counts())