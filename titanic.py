
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


df = pd.read_csv("gender_submission.csv")


print(df.shape)
print(df.head())
df.info()
print(df.describe(include="all"))
print("Duplicate rows:", df.duplicated().sum())


df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
print(df.columns.tolist())


print("\nMissing values before cleaning:")
print(df.isnull().sum())


df["age"] = df["age"].fillna(df["age"].median())
df["cabin"] = df["cabin"].fillna("Unknown")
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])


print("\nMissing values after cleaning:")
print(df.isnull().sum())


print("\nSex categories:")
print(df["sex"].unique())

print("\nEmbarked categories:")
print(df["embarked"].unique())

print("\nInvalid passenger classes:")
print(df[~df["pclass"].isin([1, 2, 3])])

print("\nInvalid survival values:")
print(df[~df["survived"].isin([0, 1])])

print("\nNegative ages:")
print(df[df["age"] < 0])

print("\nNegative fares:")
print(df[df["fare"] < 0])


print("\nMean fare:", df["fare"].mean())
print("Median fare:", df["fare"].median())
print("Mean age:", df["age"].mean())
print("Median age:", df["age"].median())
print("Most common embarked port:", df["embarked"].mode()[0])


print("\nPassenger class counts:")
print(df["pclass"].value_counts().sort_index())


for column in ["fare", "age"]:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\nColumn: {column}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)
    print("Number of potential outliers:", len(outliers))
    print(outliers[[column]])


# Visualizations — corrected to save all graphs

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)


# 1. Histogram: Fare Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["fare"].dropna(), bins=20, edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of passengers")
plt.tight_layout()
plt.savefig(plot_dir / "01_fare_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# 2. Pie chart: Passenger Distribution by Sex
counts = df["sex"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Passenger Distribution by Sex")
plt.tight_layout()
plt.savefig(plot_dir / "02_sex_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# 3. Bar chart: Survival Counts
counts = df["survived"].value_counts().reindex([0, 1], fill_value=0)

plt.figure(figsize=(6, 5))
plt.bar(["Died", "Survived"], counts.values, edgecolor="black")
plt.title("Passenger Survival Counts")
plt.xlabel("Survival status")
plt.ylabel("Number of passengers")
plt.tight_layout()
plt.savefig(plot_dir / "03_survival_counts.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# 4. Bar chart: Survival Rate by Sex
survival_rate = df.groupby("sex")["survived"].mean() * 100

plt.figure(figsize=(6, 5))
plt.bar(survival_rate.index, survival_rate.values, edgecolor="black")
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival rate (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(plot_dir / "04_survival_by_sex.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# 5. Bar chart: Survival Rate by Passenger Class
sr = df.groupby("pclass")["survived"].mean() * 100

plt.figure(figsize=(6, 5))
plt.bar(sr.index.astype(str), sr.values, edgecolor="black")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger class")
plt.ylabel("Survival rate (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(plot_dir / "05_survival_by_class.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# 6. Boxplot: Fare Distribution by Survival Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="survived", y="fare")
plt.title("Fare Distribution by Survival Status")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(plot_dir / "06_fare_boxplot_by_survival.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# 7. Correlation Heatmap
cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(plot_dir / "07_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print(f"All graphs saved successfully in: {plot_dir.resolve()}")


# Group summary
group_summary = (
    df.groupby("pclass")["survived"]
    .agg(["count", "mean", "median", "std"])
)

print("\nSurvival summary by passenger class:")
print(group_summary)


# Hypothesis testing
table = pd.crosstab(df["sex"], df["survived"])
print("\nSex vs Survival:")
print(table)

chi2, p_value, dof, expected = stats.chi2_contingency(table)

print("Chi-square statistic:", chi2)
print("Degrees of freedom:", dof)
print("P-value:", p_value)

if (expected < 5).any():
    print("Warning: Check expected cell counts for the chi-square test.")

if p_value < 0.05:
    print("Reject H0: Evidence of an association.")
else:
    print("Fail to reject H0: Insufficient evidence of an association.")


print("\nFinal dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nData types:")
print(df.dtypes)


df.to_csv("cleaned_titanic.csv", index=False)
print("Cleaned dataset saved successfully!")
