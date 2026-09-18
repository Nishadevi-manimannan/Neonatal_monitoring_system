import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ================================================================
# SETTINGS
# ================================================================

CSV_FILE = "Preprocessing_Output/preprocessed_features.csv"
OUTPUT_DIR = "Model_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("NEONATAL CRY CLASSIFICATION")
print("IMPROVED MODEL TRAINING")
print("=" * 70)

# ================================================================
# LOAD DATASET
# ================================================================

if not os.path.exists(CSV_FILE):

    print("\nERROR: Dataset not found:")
    print(CSV_FILE)
    print("\nPlace preprocessed_features.csv inside:")
    print("Preprocessing_Output")

    exit()

df = pd.read_csv(CSV_FILE)

print("\nDataset loaded successfully")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# ================================================================
# LABEL
# ================================================================

LABEL_COLUMN = "class"

if LABEL_COLUMN not in df.columns:

    print("\nERROR: 'class' column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit()

# ================================================================
# FEATURE LIST
# ================================================================

non_feature_columns = [
    "filename",
    "class",
    "original_sample_rate",
    "processed_sample_rate",
    "duration_sec"
]

feature_columns = [
    col for col in df.columns
    if col not in non_feature_columns
]

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print("\nTotal audio features:", len(feature_columns))

pitch_features = [
    x for x in feature_columns
    if x.startswith("pitch_")
]

rms_features = [
    x for x in feature_columns
    if x.startswith("rms_")
]

ste_features = [
    x for x in feature_columns
    if x.startswith("ste_")
]

zcr_features = [
    x for x in feature_columns
    if x.startswith("zcr_")
]

mfcc_features = [
    x for x in feature_columns
    if x.startswith("mfcc_")
]

gfcc_features = [
    x for x in feature_columns
    if x.startswith("gfcc_")
]

print("Pitch :", len(pitch_features))
print("RMS   :", len(rms_features))
print("STE   :", len(ste_features))
print("ZCR   :", len(zcr_features))
print("MFCC  :", len(mfcc_features))
print("GFCC  :", len(gfcc_features))

# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(df[LABEL_COLUMN].value_counts())

# ================================================================
# PREPARE X AND y
# ================================================================

X = df[feature_columns].copy()
y = df[LABEL_COLUMN].copy()

# Convert all feature values to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinity
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# ================================================================
# DATA SPLIT
# ================================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# ================================================================
# MODELS
# ================================================================

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(
            score_func=mutual_info_classif,
            k=30
        )),
        ("model", LogisticRegression(
            max_iter=3000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(
            score_func=mutual_info_classif,
            k=30
        )),
        ("model", SVC(
            C=2.0,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(
            score_func=mutual_info_classif,
            k=30
        )),
        ("model", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(
            score_func=mutual_info_classif,
            k=30
        )),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(
            score_func=mutual_info_classif,
            k=30
        )),
        ("model", ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
}

# ================================================================
# CROSS VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = []

for name, model in models.items():

    print("\n" + name)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    mean_score = scores.mean()
    std_score = scores.std()

    print(
        "Fold scores :",
        np.round(scores, 4)
    )

    print(
        "Mean F1     :",
        round(mean_score, 4)
    )

    print(
        "Std         :",
        round(std_score, 4)
    )

    cv_results.append({
        "Model": name,
        "CV_F1_Macro": mean_score,
        "CV_F1_Std": std_score
    })

# ================================================================
# TRAIN AND TEST
# ================================================================

print("\n" + "=" * 70)
print("TRAINING AND TESTING")
print("=" * 70)

test_results = []

trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )

    trained_models[name] = model

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    test_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# ================================================================
# COMBINE RESULTS
# ================================================================

cv_df = pd.DataFrame(cv_results)
test_df = pd.DataFrame(test_results)

results_df = pd.merge(
    cv_df,
    test_df,
    on="Model"
)

results_df = results_df.sort_values(
    by="CV_F1_Macro",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

# ================================================================
# SAVE RESULTS
# ================================================================

comparison_file = os.path.join(
    OUTPUT_DIR,
    "improved_model_comparison.csv"
)

try:

    results_df.to_csv(
        comparison_file,
        index=False
    )

    print("\nSaved:")
    print(comparison_file)

except PermissionError:

    comparison_file = os.path.join(
        OUTPUT_DIR,
        "improved_model_comparison_new.csv"
    )

    results_df.to_csv(
        comparison_file,
        index=False
    )

    print("\nExisting CSV was locked.")
    print("Saved new file:")
    print(comparison_file)

# ================================================================
# SELECT BEST MODEL
# ================================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print("Model:", best_model_name)

# ================================================================
# BEST MODEL PREDICTION
# ================================================================

best_predictions = best_model.predict(
    X_test
)

best_accuracy = accuracy_score(
    y_test,
    best_predictions
)

best_precision = precision_score(
    y_test,
    best_predictions,
    average="macro",
    zero_division=0
)

best_recall = recall_score(
    y_test,
    best_predictions,
    average="macro",
    zero_division=0
)

best_f1 = f1_score(
    y_test,
    best_predictions,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 70)
print("BEST MODEL RESULTS")
print("=" * 70)

print("Accuracy :", round(best_accuracy, 4))
print("Precision:", round(best_precision, 4))
print("Recall   :", round(best_recall, 4))
print("F1 Score :", round(best_f1, 4))

# ================================================================
# CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    y_test,
    best_predictions,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

report_file = os.path.join(
    OUTPUT_DIR,
    "improved_classification_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NEONATAL CRY CLASSIFICATION\n"
        "IMPROVED MODEL\n\n"
    )

    f.write(
        "Selected model: "
        + best_model_name
        + "\n\n"
    )

    f.write(report)

print("\nSaved:")
print(report_file)

# ================================================================
# CONFUSION MATRIX
# ================================================================

classes = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=classes
)

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "Confusion Matrix - " + best_model_name
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(len(classes)),
    classes,
    rotation=30
)

plt.yticks(
    range(len(classes)),
    classes
)

for i in range(len(classes)):

    for j in range(len(classes)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

cm_file = os.path.join(
    OUTPUT_DIR,
    "improved_confusion_matrix.png"
)

plt.savefig(
    cm_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(cm_file)

# ================================================================
# MODEL COMPARISON GRAPH
# ================================================================

plot_df = results_df.copy()

plt.figure(
    figsize=(10, 6)
)

x = np.arange(
    len(plot_df)
)

width = 0.2

plt.bar(
    x - 1.5 * width,
    plot_df["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    plot_df["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    plot_df["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    plot_df["F1 Score"],
    width,
    label="F1 Score"
)

plt.xticks(
    x,
    plot_df["Model"],
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Neonatal Cry Classification - Model Comparison"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()

comparison_png = os.path.join(
    OUTPUT_DIR,
    "improved_model_comparison.png"
)

plt.savefig(
    comparison_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(comparison_png)

# ================================================================
# SAVE BEST MODEL
# ================================================================

model_file = os.path.join(
    OUTPUT_DIR,
    "improved_neonatal_cry_model.joblib"
)

joblib.dump(
    best_model,
    model_file
)

print("\nSaved:")
print(model_file)

# ================================================================
# SAVE FEATURE NAMES
# ================================================================

feature_file = os.path.join(
    OUTPUT_DIR,
    "improved_feature_names.joblib"
)

joblib.dump(
    feature_columns,
    feature_file
)

print("Saved:")
print(feature_file)

# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": best_predictions
})

prediction_file = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("Saved:")
print(prediction_file)

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("IMPROVED MODEL TRAINING COMPLETE")
print("=" * 70)

print("\nDataset samples :", len(df))
print("Audio features  :", len(feature_columns))
print("Cry classes     :", len(classes))

print("\nSelected model  :", best_model_name)
print("Accuracy        :", round(best_accuracy, 4))
print("Macro F1        :", round(best_f1, 4))

print("\nAll output files are inside:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)