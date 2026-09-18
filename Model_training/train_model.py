import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. SETTINGS
# ============================================================

CSV_PATHS = [
    "preprocessed_features.csv",
    os.path.join("Preprocessing_Output", "preprocessed_features.csv"),
    os.path.join("Preprocessing_Output", "preprocessed_dataset.csv")
]

OUTPUT_DIR = "Model_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. FIND CSV
# ============================================================

csv_path = None

for path in CSV_PATHS:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    print("ERROR: CSV file not found.")
    print()
    print("Expected one of:")
    for path in CSV_PATHS:
        print(" -", path)
    print()
    input("Press Enter to exit...")
    raise SystemExit


print("=" * 70)
print("NEONATAL CRY CLASSIFICATION - MODEL TRAINING")
print("=" * 70)

print()
print("CSV file:")
print(csv_path)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(csv_path)

print()
print("Dataset loaded successfully.")
print("Rows    :", len(df))
print("Columns :", len(df))


# ============================================================
# 4. CHECK REQUIRED LABEL COLUMN
# ============================================================

label_candidates = [
    "class",
    "label",
    "cry_class",
    "Cry Class",
    "cry_class"
]

label_column = None

for column in label_candidates:
    if column in df.columns:
        label_column = column
        break

if label_column is None:
    print()
    print("ERROR: Cry class column was not found.")
    print("Available columns:")
    print(list(df.columns))
    input("Press Enter to exit...")
    raise SystemExit


print()
print("Label column:", label_column)


# ============================================================
# 5. SHOW CLASS DISTRIBUTION
# ============================================================

print()
print("-" * 70)
print("CLASS DISTRIBUTION")
print("-" * 70)

class_counts = df[label_column].value_counts()

print(class_counts)


# ============================================================
# 6. REMOVE NON-FEATURE COLUMNS
# ============================================================

metadata_columns = [
    "filename",
    "file_name",
    "audio_filename",
    "class",
    "label",
    "cry_class",
    "Cry Class",
    "original_sample_rate",
    "processed_sample_rate",
    "duration_sec",
    "duration",
    "sample_rate"
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


# Keep only numeric feature columns
numeric_feature_columns = []

for column in feature_columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        numeric_feature_columns.append(column)

feature_columns = numeric_feature_columns


print()
print("-" * 70)
print("FEATURE INFORMATION")
print("-" * 70)

print("Number of features:", len(feature_columns))

print()
print("Features:")

for i, feature in enumerate(feature_columns, start=1):
    print(f"{i:2d}. {feature}")


# ============================================================
# 7. CREATE X AND Y
# ============================================================

X = df[feature_columns].copy()
y = df[label_column].astype(str).copy()


# ============================================================
# 8. HANDLE INVALID VALUES
# ============================================================

print()
print("-" * 70)
print("DATA QUALITY CHECK")
print("-" * 70)

print("Missing values before cleaning:", X.isna().sum().sum())

# Replace infinity with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Replace missing values using column median
X = X.fillna(X.median())

print("Missing values after cleaning :", X.isna().sum().sum())


# ============================================================
# 9. ENCODE CLASS LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print()
print("-" * 70)
print("CLASS ENCODING")
print("-" * 70)

for number, class_name in enumerate(label_encoder.classes_):
    print(number, "=", class_name)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print()
print("-" * 70)
print("DATASET SPLIT")
print("-" * 70)

print("Total samples :", len(X))
print("Training      :", len(X_train))
print("Testing       :", len(X_test))


# ============================================================
# 11. DEFINE MODELS
# ============================================================

models = {

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])
}


# ============================================================
# 12. CROSS-VALIDATION ON TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = {}

for model_name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    cv_results[model_name] = scores.mean()

    print()
    print(model_name)
    print("Fold F1 scores :", np.round(scores, 4))
    print("Mean F1        :", round(scores.mean(), 4))


# ============================================================
# 13. TRAIN AND TEST EACH MODEL
# ============================================================

print()
print("=" * 70)
print("MODEL TEST RESULTS")
print("=" * 70)

results = []

trained_models = {}

for model_name, model in models.items():

    print()
    print("Training:", model_name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "CV_F1_Macro": cv_results[model_name],
        "Test_Accuracy": accuracy,
        "Test_Precision": precision,
        "Test_Recall": recall,
        "Test_F1_Macro": f1
    })

    trained_models[model_name] = model

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


# ============================================================
# 14. SAVE RESULTS CSV
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV_F1_Macro",
    ascending=False
)

results_csv = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)


print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))

print()
print("Saved:")
print(results_csv)


# ============================================================
# 15. SELECT MODEL USING CROSS-VALIDATION
# ============================================================

selected_model_name = results_df.iloc[0]["Model"]

selected_model = trained_models[selected_model_name]

print()
print("=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print(selected_model_name)
print("Selection based on highest 5-fold training CV macro-F1.")


# ============================================================
# 16. SAVE SELECTED MODEL
# ============================================================

model_file = os.path.join(
    OUTPUT_DIR,
    "neonatal_cry_model.pkl"
)

joblib.dump(
    selected_model,
    model_file
)


# ============================================================
# 17. SAVE LABEL ENCODER
# ============================================================

encoder_file = os.path.join(
    OUTPUT_DIR,
    "label_encoder.pkl"
)

joblib.dump(
    label_encoder,
    encoder_file
)


# ============================================================
# 18. SAVE FEATURE NAMES
# ============================================================

feature_file = os.path.join(
    OUTPUT_DIR,
    "feature_names.json"
)

with open(feature_file, "w", encoding="utf-8") as f:
    json.dump(
        feature_columns,
        f,
        indent=4
    )


print()
print("Saved model:")
print(model_file)

print()
print("Saved label encoder:")
print(encoder_file)

print()
print("Saved feature list:")
print(feature_file)


# ============================================================
# 19. CONFUSION MATRICES
# ============================================================

print()
print("=" * 70)
print("CREATING CONFUSION MATRICES")
print("=" * 70)

class_names = label_encoder.classes_

for model_name, model in trained_models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        model_name + " - Confusion Matrix"
    )

    plt.tight_layout()

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    graph_path = os.path.join(
        OUTPUT_DIR,
        safe_name + "_confusion_matrix.png"
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Created:", graph_path)


# ============================================================
# 20. MODEL COMPARISON GRAPH
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

x = np.arange(len(results_df))

width = 0.2

ax.bar(
    x - 1.5 * width,
    results_df["Test_Accuracy"],
    width,
    label="Accuracy"
)

ax.bar(
    x - 0.5 * width,
    results_df["Test_Precision"],
    width,
    label="Precision"
)

ax.bar(
    x + 0.5 * width,
    results_df["Test_Recall"],
    width,
    label="Recall"
)

ax.bar(
    x + 1.5 * width,
    results_df["Test_F1_Macro"],
    width,
    label="F1 Score"
)

ax.set_xticks(x)
ax.set_xticklabels(
    results_df["Model"],
    rotation=20
)

ax.set_ylim(0, 1.05)

ax.set_ylabel("Score")

ax.set_title(
    "Neonatal Cry Classification - Model Comparison"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

comparison_graph = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

plt.savefig(
    comparison_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Created:", comparison_graph)


# ============================================================
# 21. CLASSIFICATION REPORT FOR SELECTED MODEL
# ============================================================

y_pred_selected = selected_model.predict(X_test)

report = classification_report(
    y_test,
    y_pred_selected,
    target_names=class_names,
    zero_division=0
)

report_file = os.path.join(
    OUTPUT_DIR,
    "selected_model_classification_report.txt"
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
        "Selected Model: "
        + selected_model_name
        + "\n\n"
    )

    f.write(report)


print()
print("Classification report:")
print(report)

print()
print("Saved:")
print(report_file)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print()
print("Dataset samples :", len(df))
print("Audio features  :", len(feature_columns))
print("Cry classes     :", len(class_names))

print()
print("Classes:")

for class_name in class_names:
    print(" -", class_name)

print()
print("Selected model :", selected_model_name)

print()
print("All output files are inside:")
print(OUTPUT_DIR)

print()
print("=" * 70)