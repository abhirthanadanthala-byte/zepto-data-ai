# ============================================================
# MODULE 2 - ANALYTICS
# PART A - EXPLORATORY DATA ANALYSIS
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. FOLDERS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TITANIC_CSV = os.path.join(BASE_DIR, "titanic.csv")


# ------------------------------------------------------------
# 2. LOAD TITANIC DATASET ONCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. LOADING TITANIC DATASET")
print("=" * 70)

df = sns.load_dataset("titanic")

# Save immediately as required
df.to_csv(TITANIC_CSV, index=False)

print(f"Titanic dataset saved to: {TITANIC_CSV}")

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 3. BASIC DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. DATASET INFORMATION")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nInfo:")
df.info()

print("\nDescribe:")
print(df.describe(include="all").T)


# ------------------------------------------------------------
# 4. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()
missing_percent = (df.isnull().mean() * 100).round(2)

missing_table = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percent
})

print(missing_table)

missing_table.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values.csv")
)


# ------------------------------------------------------------
# 5. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. MISSING VALUE HANDLING")
print("=" * 70)

df_clean = df.copy()

# Calculate percentages before modification
missing_pct = df_clean.isnull().mean() * 100

# Columns with < 5% missing -> drop rows
low_missing_cols = [
    col for col in df_clean.columns
    if 0 < missing_pct[col] < 5
]

print("\nColumns with <5% missing:")
print(low_missing_cols)

if low_missing_cols:
    df_clean = df_clean.dropna(subset=low_missing_cols)


# Columns with 5-30% missing -> impute
medium_missing_cols = [
    col for col in df_clean.columns
    if 5 <= missing_pct[col] <= 30
]

print("\nColumns with 5-30% missing:")
print(medium_missing_cols)

# Age -> median
if "age" in medium_missing_cols:
    df_clean["age"] = df_clean["age"].fillna(
        df_clean["age"].median()
    )

# Embarked -> mode
if "embarked" in medium_missing_cols:
    df_clean["embarked"] = df_clean["embarked"].fillna(
        df_clean["embarked"].mode()[0]
    )


# Very high missing columns
high_missing_cols = [
    col for col in df_clean.columns
    if missing_pct[col] > 30
]

print("\nColumns with >30% missing:")
print(high_missing_cols)

# For Titanic, deck has very high missingness.
# Drop it because it contains more than 30% missing values.
for col in high_missing_cols:
    if col in df_clean.columns:
        df_clean.drop(columns=[col], inplace=True)
        print(f"Dropped high-missing column: {col}")


print("\nRemaining missing values:")
print(df_clean.isnull().sum())


# Save cleaned dataset
CLEAN_CSV = os.path.join(OUTPUT_DIR, "titanic_cleaned.csv")
df_clean.to_csv(CLEAN_CSV, index=False)

print(f"\nCleaned dataset saved to: {CLEAN_CSV}")


# ------------------------------------------------------------
# 6. AGE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. AGE ANALYSIS")
print("=" * 70)

age = df_clean["age"].dropna()

Q1_age = age.quantile(0.25)
Q3_age = age.quantile(0.75)
IQR_age = Q3_age - Q1_age

lower_age = Q1_age - 1.5 * IQR_age
upper_age = Q3_age + 1.5 * IQR_age

age_outliers = age[
    (age < lower_age) |
    (age > upper_age)
]

print(f"Age Q1: {Q1_age:.2f}")
print(f"Age Q3: {Q3_age:.2f}")
print(f"Age IQR: {IQR_age:.2f}")
print(f"Age lower bound: {lower_age:.2f}")
print(f"Age upper bound: {upper_age:.2f}")
print(f"Age outlier count: {len(age_outliers)}")


# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(age, bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "age_histogram.png"),
    dpi=150
)
plt.close()


# Boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=age)
plt.title("Age Boxplot")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "age_boxplot.png"),
    dpi=150
)
plt.close()


# ------------------------------------------------------------
# 7. FARE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. FARE ANALYSIS")
print("=" * 70)

fare = df_clean["fare"].dropna()

Q1_fare = fare.quantile(0.25)
Q3_fare = fare.quantile(0.75)
IQR_fare = Q3_fare - Q1_fare

lower_fare = Q1_fare - 1.5 * IQR_fare
upper_fare = Q3_fare + 1.5 * IQR_fare

fare_outliers = fare[
    (fare < lower_fare) |
    (fare > upper_fare)
]

fare_mean = fare.mean()
fare_median = fare.median()
fare_mode = fare.mode()[0]

print(f"Fare mean: {fare_mean:.2f}")
print(f"Fare median: {fare_median:.2f}")
print(f"Fare mode: {fare_mode:.2f}")

print(f"Fare Q1: {Q1_fare:.2f}")
print(f"Fare Q3: {Q3_fare:.2f}")
print(f"Fare IQR: {IQR_fare:.2f}")
print(f"Fare lower bound: {lower_fare:.2f}")
print(f"Fare upper bound: {upper_fare:.2f}")
print(f"Fare outlier count: {len(fare_outliers)}")

if fare_mean > fare_median:
    fare_skew = "Right-skewed"
elif fare_mean < fare_median:
    fare_skew = "Left-skewed"
else:
    fare_skew = "Approximately symmetric"

print(f"Fare skew direction: {fare_skew}")


# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(fare, bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "fare_histogram.png"),
    dpi=150
)
plt.close()


# Boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=fare)
plt.title("Fare Boxplot")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "fare_boxplot.png"),
    dpi=150
)
plt.close()


# ------------------------------------------------------------
# 8. SURVIVAL RATE BY SEX
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("7. SURVIVAL RATE BY SEX")
print("=" * 70)

sex_survival = (
    df_clean.groupby("sex")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(sex_survival)


# Boolean masking demonstration
male_survival = (
    df_clean.loc[df_clean["sex"] == "male", "survived"].mean()
    * 100
)

female_survival = (
    df_clean.loc[df_clean["sex"] == "female", "survived"].mean()
    * 100
)

print(f"\nMale survival rate: {male_survival:.2f}%")
print(f"Female survival rate: {female_survival:.2f}%")


# ------------------------------------------------------------
# 9. SURVIVAL RATE BY PCLASS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("8. SURVIVAL RATE BY PCLASS")
print("=" * 70)

pclass_survival = (
    df_clean.groupby("pclass")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(pclass_survival)


# Boolean masking
for pclass in sorted(df_clean["pclass"].unique()):
    rate = (
        df_clean.loc[
            df_clean["pclass"] == pclass,
            "survived"
        ].mean() * 100
    )

    print(f"Class {pclass}: {rate:.2f}%")


# ------------------------------------------------------------
# 10. SURVIVAL BY SEX + PCLASS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("9. SURVIVAL RATE BY SEX + PCLASS")
print("=" * 70)

sex_pclass_survival = (
    df_clean
    .groupby(["sex", "pclass"])["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(sex_pclass_survival)


# ------------------------------------------------------------
# 11. EXACT SIX-COLUMN CORRELATION MATRIX
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10. CORRELATION MATRIX")
print("=" * 70)

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df_clean[corr_columns].corr()

print(corr_matrix.round(3))

corr_matrix.to_csv(
    os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
)


# Heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Titanic Correlation Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 12. FIND TWO STRONGEST CORRELATIONS
# ------------------------------------------------------------

corr_pairs = []

for i in range(len(corr_columns)):
    for j in range(i + 1, len(corr_columns)):

        col1 = corr_columns[i]
        col2 = corr_columns[j]

        value = corr_matrix.loc[col1, col2]

        corr_pairs.append(
            (col1, col2, value, abs(value))
        )

corr_pairs = sorted(
    corr_pairs,
    key=lambda x: x[3],
    reverse=True
)

print("\nTop 2 strongest absolute correlations:")

for pair in corr_pairs[:2]:
    print(
        f"{pair[0]} <-> {pair[1]} : "
        f"{pair[2]:.3f}"
    )


# ------------------------------------------------------------
# 13. MULTIVARIATE CHART 1
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "multivariate_1_survival_class_sex.png"),
    dpi=150
)

plt.close()

print("\nChart 1 interpretation:")
print(
    "The chart compares survival across passenger classes and sex. "
    "Survival rates differ across classes, while sex also shows "
    "a clear relationship with survival."
)


# ------------------------------------------------------------
# 14. MULTIVARIATE CHART 2
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x="pclass",
    y="fare",
    hue="sex"
)

plt.title("Fare Distribution by Class and Sex")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "multivariate_2_fare_class_sex.png"),
    dpi=150
)

plt.close()

print("\nChart 2 interpretation:")
print(
    "Fare distributions vary considerably between passenger classes. "
    "Higher-class passengers generally paid higher fares, and the "
    "distribution also differs between male and female passengers."
)


# ------------------------------------------------------------
# 15. MULTIVARIATE CHART 3
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df_clean,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7
)

plt.title("Age vs Fare by Survival and Sex")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "multivariate_3_age_fare_survival.png"),
    dpi=150
)

plt.close()

print("\nChart 3 interpretation:")
print(
    "The scatter plot shows the relationship between passenger age "
    "and fare while distinguishing survival and sex. Fare values are "
    "more dispersed for some passengers, particularly at higher fare levels."
)


# ------------------------------------------------------------
# 16. MULTIVARIATE CHART 4
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.violinplot(
    data=df_clean,
    x="pclass",
    y="age",
    hue="survived",
    split=True
)

plt.title("Age Distribution by Class and Survival")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "multivariate_4_age_class_survival.png"),
    dpi=150
)

plt.close()

print("\nChart 4 interpretation:")
print(
    "Age distributions vary across passenger classes and survival "
    "groups. The plot allows the spread and central tendency of age "
    "to be compared simultaneously across these categories."
)


# ------------------------------------------------------------
# 17. EDA-ONLY STANDARDIZATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("11. EDA-ONLY STANDARDIZATION")
print("=" * 70)

standardized_df = df_clean.copy()

for column in ["age", "fare"]:

    mean_before = standardized_df[column].mean()
    std_before = standardized_df[column].std()

    standardized_df[column + "_z"] = (
        standardized_df[column] - mean_before
    ) / std_before

    print(f"\n{column.upper()} BEFORE:")
    print(f"Mean = {mean_before:.4f}")
    print(f"Std = {std_before:.4f}")

    print(f"{column.upper()} AFTER:")
    print(
        f"Mean = "
        f"{standardized_df[column + '_z'].mean():.4f}"
    )

    print(
        f"Std = "
        f"{standardized_df[column + '_z'].std():.4f}"
    )


standardized_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "eda_standardized_data.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 18. SAVE EDA SUMMARY
# ------------------------------------------------------------

summary_file = os.path.join(
    OUTPUT_DIR,
    "eda_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("TITANIC DATASET - EDA SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Original shape: {df.shape}\n")
    f.write(f"Cleaned shape: {df_clean.shape}\n\n")

    f.write("Missing values before cleaning:\n")
    f.write(missing_table.to_string())
    f.write("\n\n")

    f.write(f"Age outlier count: {len(age_outliers)}\n")
    f.write(f"Fare outlier count: {len(fare_outliers)}\n")
    f.write(f"Fare mean: {fare_mean:.4f}\n")
    f.write(f"Fare median: {fare_median:.4f}\n")
    f.write(f"Fare mode: {fare_mode:.4f}\n")
    f.write(f"Fare skew direction: {fare_skew}\n\n")

    f.write("Survival rate by sex:\n")
    f.write(sex_survival.to_string())
    f.write("\n\n")

    f.write("Survival rate by passenger class:\n")
    f.write(pclass_survival.to_string())
    f.write("\n\n")

    f.write("Survival rate by sex and passenger class:\n")
    f.write(sex_pclass_survival.to_string())
    f.write("\n\n")

    f.write("Correlation matrix:\n")
    f.write(corr_matrix.round(3).to_string())
    f.write("\n\n")

    f.write("Top two absolute correlations:\n")

    for pair in corr_pairs[:2]:
        f.write(
            f"{pair[0]} <-> {pair[1]} = "
            f"{pair[2]:.3f}\n"
        )

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"\nOutputs saved inside:")
print(OUTPUT_DIR)

print("\nGenerated files include:")
print("- titanic.csv")
print("- outputs/missing_values.csv")
print("- outputs/titanic_cleaned.csv")
print("- outputs/eda_summary.txt")
print("- outputs/correlation_matrix.csv")
print("- outputs/correlation_heatmap.png")
print("- outputs/age_histogram.png")
print("- outputs/age_boxplot.png")
print("- outputs/fare_histogram.png")
print("- outputs/fare_boxplot.png")
print("- outputs/multivariate charts")
print("- outputs/eda_standardized_data.csv")