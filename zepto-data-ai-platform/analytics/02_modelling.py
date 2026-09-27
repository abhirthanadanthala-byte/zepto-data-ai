# ============================================================
# MODULE 2 - PART B
# MACHINE LEARNING MODELING
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = BASE_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = BASE_DIR / "titanic.csv"

RANDOM_STATE = 42


# ============================================================
# HELPER - ONE HOT ENCODER
# ============================================================

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


# ============================================================
# 1. LOAD THE COMMITTED TITANIC CSV
# ============================================================

print("\n" + "=" * 70)
print("1. LOADING TITANIC DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("Dataset loaded from:")
print(DATA_FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CLASSIFICATION TARGET
# ============================================================

TARGET = "survived"

# We deliberately exclude:
# survived -> target
# alive -> direct duplicate of target
# who/adult_male/alone -> derived/redundant variables

classification_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "sex",
    "embarked",
    "class"
]

X = df[classification_features].copy()
y = df[TARGET].copy()


# ============================================================
# 2. STRATIFIED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("2. STRATIFIED TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nOverall class distribution:")
print(y.value_counts(normalize=True))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))

print(
    "\nStratification is used so that the proportion of survived "
    "and non-survived passengers remains approximately consistent "
    "between training and testing data."
)


# ============================================================
# 3. TRAIN-ONLY PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("3. TRAIN-ONLY PREPROCESSING")
print("=" * 70)

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked",
    "class"
]


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            encoder
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


print("Numeric preprocessing:")
print("Missing values -> median")
print("Scaling -> StandardScaler")

print("\nCategorical preprocessing:")
print("Missing values -> most frequent")
print("Encoding -> OneHotEncoder")


# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ============================================================
# 5. TRAIN MODELS + EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("5. CLASSIFICATION MODELS")
print("=" * 70)


classification_results = []
trained_pipelines = {}


def evaluate_classifier(name, model_pipeline):

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    probabilities = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        PLOTS_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=150
    )

    plt.close()

    classification_results.append(
        {
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }
    )

    return model_pipeline


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )

    trained_pipelines[name] = evaluate_classifier(
        name,
        pipeline
    )


classification_df = pd.DataFrame(
    classification_results
)

print("\n" + "=" * 70)
print("CLASSIFICATION COMPARISON")
print("=" * 70)

print(
    classification_df.to_string(
        index=False
    )
)

classification_df.to_csv(
    OUTPUT_DIR / "classification_metrics.csv",
    index=False
)


# ============================================================
# 6. DECISION TREE VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("6. DECISION TREE VISUALIZATION")
print("=" * 70)

tree_pipeline = trained_pipelines["Decision Tree"]

tree_model = tree_pipeline.named_steps["classifier"]

tree_preprocessor = tree_pipeline.named_steps["preprocessor"]

feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)

plt.figure(figsize=(24, 14))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=[
        "Did Not Survive",
        "Survived"
    ],
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=8
)

plt.title(
    "Decision Tree Classifier"
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "decision_tree.png",
    dpi=150
)

plt.close()

print("Decision tree saved.")


# ============================================================
# 7. CLASS IMBALANCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. CLASS IMBALANCE ANALYSIS")
print("=" * 70)

print("\nClass counts:")
print(y.value_counts())

print("\nClass percentages:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ------------------------------------------------------------
# 7A. BASELINE
# ------------------------------------------------------------

baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ------------------------------------------------------------
# 7B. CLASS WEIGHT BALANCED
# ------------------------------------------------------------

balanced_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ------------------------------------------------------------
# 7C. SMOTE
# ------------------------------------------------------------

smote_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)


imbalance_pipelines = {
    "Baseline": baseline_pipeline,
    "Class Weight Balanced": balanced_pipeline,
    "SMOTE": smote_pipeline
}


imbalance_results = []


for name, pipeline in imbalance_pipelines.items():

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    imbalance_results.append(
        {
            "variant": name,
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        }
    )


imbalance_df = pd.DataFrame(
    imbalance_results
)

print("\nImbalance comparison:")
print(
    imbalance_df.to_string(
        index=False
    )
)

imbalance_df.to_csv(
    OUTPUT_DIR / "imbalance_comparison.csv",
    index=False
)


# ============================================================
# 8. RANDOM FOREST GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("8. RANDOM FOREST GRID SEARCH")
print("=" * 70)


rf_base = RandomForestClassifier(
    random_state=RANDOM_STATE,
    oob_score=True,
    bootstrap=True,
    n_jobs=-1
)


rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            rf_base
        )
    ]
)


param_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],

    "classifier__max_depth": [
        None,
        5,
        10
    ],

    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True
)


grid_search.fit(
    X_train,
    y_train
)


print("\nBest parameters:")
print(grid_search.best_params_)

print(
    "\nBest cross-validation F1:",
    round(grid_search.best_score_, 4)
)


grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results.to_csv(
    OUTPUT_DIR / "rf_grid_search_results.csv",
    index=False
)


# ============================================================
# 9. FINAL RANDOM FOREST WITH OOB SCORE
# ============================================================

print("\n" + "=" * 70)
print("9. FINAL RANDOM FOREST")
print("=" * 70)


best_params = grid_search.best_params_

best_n_estimators = best_params[
    "classifier__n_estimators"
]

best_max_depth = best_params[
    "classifier__max_depth"
]

best_max_features = best_params[
    "classifier__max_features"
]


final_rf = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    max_features=best_max_features,
    random_state=RANDOM_STATE,
    oob_score=True,
    bootstrap=True,
    n_jobs=-1
)


final_rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            final_rf
        )
    ]
)


final_rf_pipeline.fit(
    X_train,
    y_train
)


final_rf_model = (
    final_rf_pipeline
    .named_steps["classifier"]
)


print(
    "Final RF OOB Score:",
    round(
        final_rf_model.oob_score_,
        4
    )
)


final_predictions = final_rf_pipeline.predict(
    X_test
)

final_probabilities = (
    final_rf_pipeline
    .predict_proba(X_test)[:, 1]
)


final_rf_metrics = {
    "accuracy": accuracy_score(
        y_test,
        final_predictions
    ),
    "precision": precision_score(
        y_test,
        final_predictions,
        zero_division=0
    ),
    "recall": recall_score(
        y_test,
        final_predictions,
        zero_division=0
    ),
    "f1": f1_score(
        y_test,
        final_predictions,
        zero_division=0
    ),
    "roc_auc": roc_auc_score(
        y_test,
        final_probabilities
    ),
    "oob_score": final_rf_model.oob_score_
}


print("\nFinal Random Forest metrics:")

for key, value in final_rf_metrics.items():
    print(
        f"{key}: {value:.4f}"
    )


pd.DataFrame(
    [final_rf_metrics]
).to_csv(
    OUTPUT_DIR / "final_random_forest_metrics.csv",
    index=False
)


# ============================================================
# 10. SAVE COMPLETE FITTED PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("10. SAVING COMPLETE PIPELINE")
print("=" * 70)


MODEL_FILE = (
    OUTPUT_DIR /
    "best_rf_pipeline.joblib"
)


joblib.dump(
    final_rf_pipeline,
    MODEL_FILE
)


print(
    "Pipeline saved to:",
    MODEL_FILE
)


# ============================================================
# 11. RELOAD PIPELINE AND PREDICT RAW INPUT
# ============================================================

print("\n" + "=" * 70)
print("11. RELOAD SAVED PIPELINE")
print("=" * 70)


loaded_pipeline = joblib.load(
    MODEL_FILE
)


sample_raw_input = X_test.iloc[:3].copy()

sample_predictions = (
    loaded_pipeline
    .predict(sample_raw_input)
)


print("\nRaw input:")
print(sample_raw_input)

print("\nPredictions:")
print(sample_predictions)


# ============================================================
# 12. REGRESSION - PREDICT FARE
# ============================================================

print("\n" + "=" * 70)
print("12. REGRESSION - PREDICT FARE")
print("=" * 70)


regression_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked",
    "class"
]

X_reg = df[regression_features].copy()

y_reg = df["fare"].copy()


X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE
)


reg_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked",
    "class"
]


reg_numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


reg_categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            encoder
        )
    ]
)


reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_transformer,
            reg_numeric_features
        ),
        (
            "categorical",
            reg_categorical_transformer,
            reg_categorical_features
        )
    ]
)


regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


reg_predictions = regression_pipeline.predict(
    X_reg_test
)


# ============================================================
# 13. REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_reg_test,
    reg_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        reg_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    reg_predictions
)


reg_feature_count = len(
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

n_test = len(
    y_reg_test
)

if n_test > reg_feature_count + 1:

    adjusted_r2 = (
        1
        -
        (1 - r2)
        *
        (n_test - 1)
        /
        (n_test - reg_feature_count - 1)
    )

else:

    adjusted_r2 = np.nan


print("\nRegression metrics:")

print(
    "MAE:",
    round(mae, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "R²:",
    round(r2, 4)
)

print(
    "Adjusted R²:",
    round(adjusted_r2, 4)
)


regression_metrics_df = pd.DataFrame(
    [
        {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2
        }
    ]
)


regression_metrics_df.to_csv(
    OUTPUT_DIR / "regression_metrics.csv",
    index=False
)


# ============================================================
# 14. RESIDUAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("14. RESIDUAL ANALYSIS")
print("=" * 70)


residuals = (
    y_reg_test.values
    -
    reg_predictions
)


plt.figure(figsize=(8, 6))

plt.scatter(
    reg_predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot - Fare Regression"
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "fare_regression_residuals.png",
    dpi=150
)

plt.close()


# Simple heteroscedasticity diagnostic
absolute_residual_correlation = np.corrcoef(
    reg_predictions,
    np.abs(residuals)
)[0, 1]


print(
    "Correlation between predicted fare and absolute residuals:",
    round(
        absolute_residual_correlation,
        4
    )
)


if abs(absolute_residual_correlation) >= 0.20:

    hetero_conclusion = (
        "There is some evidence of heteroscedasticity because "
        "the magnitude of residuals changes with predicted fare."
    )

else:

    hetero_conclusion = (
        "There is no strong evidence of heteroscedasticity "
        "from the residual magnitude diagnostic."
    )


print("\nHeteroscedasticity conclusion:")
print(hetero_conclusion)


with open(
    OUTPUT_DIR / "heteroscedasticity_conclusion.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        hetero_conclusion
        +
        "\n\n"
        +
        f"Correlation between predicted fare and "
        f"absolute residuals: "
        f"{absolute_residual_correlation:.4f}\n"
    )


# ============================================================
# 15. FINAL MODEL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("15. FINAL MODEL RECOMMENDATION")
print("=" * 70)


classification_df_sorted = classification_df.sort_values(
    by="f1",
    ascending=False
)

recommended_model = classification_df_sorted.iloc[0]


recommendation = f"""
Classification model comparison:

The {recommended_model['model']} produced an F1 score of
{recommended_model['f1']:.4f} and a ROC-AUC of
{recommended_model['roc_auc']:.4f} on the held-out test set.

Accuracy was {recommended_model['accuracy']:.4f}, while precision
was {recommended_model['precision']:.4f} and recall was
{recommended_model['recall']:.4f}.

The final Random Forest was also evaluated using its out-of-bag
score, which was {final_rf_model.oob_score_:.4f}.

For the regression task, the fare prediction model obtained an
MAE of {mae:.4f}, RMSE of {rmse:.4f}, R² of {r2:.4f}, and
Adjusted R² of {adjusted_r2:.4f}.
"""


print(
    recommendation
)


with open(
    OUTPUT_DIR / "final_model_recommendation.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        recommendation.strip()
    )


# ============================================================
# 16. SAVE MODELING SUMMARY
# ============================================================

summary = f"""
MODULE 2 - MACHINE LEARNING SUMMARY
====================================

Dataset:
Titanic dataset loaded from analytics/titanic.csv

Classification target:
survived

Classification features:
{classification_features}

Train/test split:
80/20 stratified split

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

Classification metrics:
{classification_df.to_string(index=False)}

Random Forest GridSearchCV:
Best parameters:
{grid_search.best_params_}

Best CV F1:
{grid_search.best_score_:.4f}

Final Random Forest OOB score:
{final_rf_model.oob_score_:.4f}

Regression target:
fare

Regression metrics:
MAE = {mae:.4f}
RMSE = {rmse:.4f}
R2 = {r2:.4f}
Adjusted R2 = {adjusted_r2:.4f}

Heteroscedasticity:
{hetero_conclusion}

Saved model:
{MODEL_FILE}
"""


with open(
    OUTPUT_DIR / "modeling_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(summary)


print("\n" + "=" * 70)
print("MODULE 2 PART B COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print(" - classification_metrics.csv")
print(" - imbalance_comparison.csv")
print(" - rf_grid_search_results.csv")
print(" - final_random_forest_metrics.csv")
print(" - regression_metrics.csv")
print(" - heteroscedasticity_conclusion.txt")
print(" - final_model_recommendation.txt")
print(" - modeling_summary.txt")
print(" - best_rf_pipeline.joblib")

print("\nGenerated plots:")
print(" - Decision tree")
print(" - Confusion matrices")
print(" - Fare regression residual plot")