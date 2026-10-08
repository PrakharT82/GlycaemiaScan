import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("reports/figures").mkdir(parents=True, exist_ok=True)

# Load and inspect
df = pd.read_csv("data/diabetes.csv")
print(df.shape)
print(df.head())
df.info()
print(df.describe())

# Hidden missing values: zeros that are biologically impossible
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\nZeros per column:\n", (df[cols] == 0).sum())
df[cols] = df[cols].replace(0, np.nan)
print("\nMissing after conversion:\n", df.isna().sum())

# Class balance
print("\nClass balance:\n", df["Outcome"].value_counts(normalize=True))
df["Outcome"].value_counts().plot(kind="bar", color=["#4C9F70", "#D9534F"])
plt.title("Class balance (0 = No diabetes, 1 = Diabetes)")
plt.savefig("reports/figures/class_balance.png", bbox_inches="tight")
plt.close()

# Histograms
df.hist(figsize=(12, 10), bins=20)
plt.tight_layout()
plt.savefig("reports/figures/histograms.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.savefig("reports/figures/correlation_heatmap.png", bbox_inches="tight")
plt.close()

# Box plots
features = df.columns.drop("Outcome")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.ravel(), features):
    sns.boxplot(x="Outcome", y=col, data=df, ax=ax)
plt.tight_layout()
plt.savefig("reports/figures/boxplots.png")
plt.close()

# IQR outlier count (reported but kept: extreme medical values are often real)
for col in features:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"{col}: {n} outliers")