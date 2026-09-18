import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ================================================================
# SETTINGS
# ================================================================

CSV_FILE = os.path.join(
    "Preprocessing_Output",
    "preprocessed_features.csv"
)

OUTPUT_DIR = "Model_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ================================================================
# LOAD DATA
# ================================================================

print("=" * 70)
print("NEONATAL CRY CLASSIFICATION")
print("FAST SVM TUNING")
print("=" * 70)

df = pd.read_csv(CSV_FILE)

print()
print("Dataset loaded successfully")
print("Rows    :", len(df))
print("Columns :", len(df))

# ================================================================
# FEATURES
# ================================================================

metadata = [
    "filename",
    "class",
    "original_sample_rate",
    "processed_sample_rate",
    "duration_sec"
]

feature_columns = [
    c for c in df.columns
    if c not in metadata
]

X = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

y = df["class"]

print()
print("Audio features:", len(feature_columns))
print("Samples       :", len(df))

# ================================================================
# SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print()
print("Training:", len(X_train))
print("Testing :", len(X_test))

# ================================================================
# SVM PIPELINE
# ================================================================

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svm",
        SVC(
            probability=True,
            random_state=RANDOM_STATE
        )
    )
])

# ================================================================
# SMALL PARAMETER GRID
# ================================================================

param_grid = {
    "svm__C": [
        0.1,
        1,
        10
    ],

    "svm__gamma": [
        "scale",
        0.01
    ],

    "svm__kernel": [
        "rbf",
        "linear"
    ],

    "svm__class_weight": [
        None,
        "balanced"
    ]
}

# 3 × 2 × 2 × 2 = 24 candidates
# 24 × 5 folds = 120 fits

print()
print("=" * 70)
print("FAST SVM HYPERPARAMETER TUNING")
print("=" * 70)

print()
print("Candidates : 24")
print("CV folds   : 5")
print("Total fits : 120")
print()
print("Starting tuning...")

# ================================================================
# CROSS VALIDATION
# ================================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(
    X_train,
    y_train
)

# ================================================================
# BEST PARAMETERS
# ================================================================

print()
print("=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print()

for key, value in grid.best_params_.items():
    print(key, "=", value)

print()
print(
    "Best CV Macro-F1:",
    round(grid.best_score_, 4)
)

# ================================================================
# TEST
# ================================================================

best_model = grid.best_estimator_

pred = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    pred,
    average="macro",
    zero_division=0
)

print()
print("=" * 70)
print("TUNED SVM TEST RESULTS")
print("=" * 70)

print()
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    pred,
    zero_division=0
)

print()
print(report)

# ================================================================
# SAVE REPORT
# ================================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "tuned_svm_classification_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NEONATAL CRY CLASSIFICATION\n"
    )

    f.write(
        "TUNED SVM MODEL\n\n"
    )

    f.write(
        "Dataset samples: "
        + str(len(df))
        + "\n"
    )

    f.write(
        "Audio features: "
        + str(len(feature_columns))
        + "\n"
    )

    f.write(
        "Training samples: "
        + str(len(X_train))
        + "\n"
    )

    f.write(
        "Testing samples: "
        + str(len(X_test))
        + "\n\n"
    )

    f.write("BEST PARAMETERS\n")

    for key, value in grid.best_params_.items():
        f.write(
            str(key)
            + " = "
            + str(value)
            + "\n"
        )

    f.write(
        "\nCV Macro-F1: "
        + str(round(grid.best_score_, 4))
        + "\n\n"
    )

    f.write("TEST RESULTS\n")

    f.write(
        "Accuracy: "
        + str(round(accuracy, 4))
        + "\n"
    )

    f.write(
        "Precision: "
        + str(round(precision, 4))
        + "\n"
    )

    f.write(
        "Recall: "
        + str(round(recall, 4))
        + "\n"
    )

    f.write(
        "F1 Score: "
        + str(round(f1, 4))
        + "\n\n"
    )

    f.write("CLASSIFICATION REPORT\n\n")
    f.write(report)

# ================================================================
# CONFUSION MATRIX CSV
# ================================================================

classes = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    pred,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

cm_file = os.path.join(
    OUTPUT_DIR,
    "tuned_svm_confusion_matrix.csv"
)

cm_df.to_csv(cm_file)

# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame({
    "Actual_Class": y_test.values,
    "Predicted_Class": pred
})

prediction_file = os.path.join(
    OUTPUT_DIR,
    "tuned_svm_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

# ================================================================
# SAVE MODEL
# ================================================================

model_file = os.path.join(
    OUTPUT_DIR,
    "final_svm_neonatal_cry_model.joblib"
)

joblib.dump(
    best_model,
    model_file
)

# ================================================================
# SAVE FEATURE NAMES
# ================================================================

feature_file = os.path.join(
    OUTPUT_DIR,
    "final_svm_feature_names.joblib"
)

joblib.dump(
    feature_columns,
    feature_file
)

# ================================================================
# SAVE TUNING RESULTS
# ================================================================

results = pd.DataFrame(
    grid.cv_results_
)

results = results[
    [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_svm__C",
        "param_svm__gamma",
        "param_svm__kernel",
        "param_svm__class_weight"
    ]
]

results = results.sort_values(
    "rank_test_score"
)

results_file = os.path.join(
    OUTPUT_DIR,
    "svm_tuning_results.csv"
)

results.to_csv(
    results_file,
    index=False
)

# ================================================================
# COMPLETE
# ================================================================

print()
print("=" * 70)
print("SVM TUNING COMPLETE")
print("=" * 70)

print()
print("Final model      : Tuned SVM")
print("CV Macro-F1      :", round(grid.best_score_, 4))
print("Test Accuracy    :", round(accuracy, 4))
print("Test Precision   :", round(precision, 4))
print("Test Recall      :", round(recall, 4))
print("Test F1 Score    :", round(f1, 4))

print()
print("Saved files:")
print(" - final_svm_neonatal_cry_model.joblib")
print(" - final_svm_feature_names.joblib")
print(" - tuned_svm_classification_report.txt")
print(" - tuned_svm_confusion_matrix.csv")
print(" - tuned_svm_test_predictions.csv")
print(" - svm_tuning_results.csv")

print()
print("=" * 70)