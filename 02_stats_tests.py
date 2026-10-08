import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_csv("data/diabetes.csv")
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)

# Mann-Whitney U: do diabetic and non-diabetic groups differ?
# H0: both groups come from the same distribution
rows = []
for col in df.columns.drop("Outcome"):
    diabetic = df.loc[df["Outcome"] == 1, col].dropna()
    healthy = df.loc[df["Outcome"] == 0, col].dropna()
    stat, p = mannwhitneyu(diabetic, healthy)
    rows.append({
        "Feature": col,
        "Median_Diabetic": diabetic.median(),
        "Median_NonDiabetic": healthy.median(),
        "p_value": p,
        "Significant": bool(p < 0.05),
    })

result = pd.DataFrame(rows).sort_values("p_value")
print(result.to_string(index=False))
result.to_csv("reports/stat_tests.csv", index=False)