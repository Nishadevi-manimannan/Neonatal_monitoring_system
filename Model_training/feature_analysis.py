import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# NEONATAL CRY DATASET - FEATURE ANALYSIS
# ============================================================

CSV_FILE = os.path.join(
    "Preprocessing_Output",
    "preprocessed_features.csv"
)

OUTPUT_DIR = os.path.join(
    "Preprocessing_Output",
    "Feature_Analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("NEONATAL CRY DATASET - FEATURE ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

if not os.path.exists(CSV_FILE):
    print()
    print("ERROR: CSV file not found:")
    print(CSV_FILE)
    print()
    print("Make sure preprocessing was completed first.")
    raise SystemExit

df = pd.read_csv(CSV_FILE)

print()
print("CSV loaded successfully.")
print("Samples :", len(df))
print("Columns :", len(df))

# ------------------------------------------------------------
# BASIC DATASET INFORMATION
# ------------------------------------------------------------

print()
print("-" * 70)
print("CLASS DISTRIBUTION")
print("-" * 70)

class_counts = df["class"].value_counts()

print(class_counts)

# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print()
print("-" * 70)
print("MISSING VALUE CHECK")
print("-" * 70)

missing = df.isnull().sum()

total_missing = missing.sum()

if total_missing == 0:
    print("No missing values found.")
else:
    print("Missing values found:")
    print(missing[missing > 0])

# ------------------------------------------------------------
# DUPLICATE CHECK
# ------------------------------------------------------------

print()
print("-" * 70)
print("DUPLICATE CHECK")
print("-" * 70)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

# ------------------------------------------------------------
# CLASS DISTRIBUTION GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

class_counts.plot(
    kind="bar"
)

plt.title(
    "Cry Class Distribution"
)

plt.xlabel(
    "Cry Class"
)

plt.ylabel(
    "Number of Audio Samples"
)

plt.xticks(
    rotation=0
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_class_distribution.svg"
    ),
    format="svg"
)

plt.close()

print()
print("Created: 01_class_distribution.svg")


# ============================================================
# FEATURE COMPARISON FUNCTION
# ============================================================

def comparison_graph(
    column,
    title,
    ylabel,
    filename
):

    if column not in df.columns:
        print(
            "Column not found:",
            column
        )
        return

    classes = df["class"].unique()

    data = []

    labels = []

    for class_name in classes:

        values = df.loc[
            df["class"] == class_name,
            column
        ].dropna()

        if len(values) > 0:

            data.append(
                values
            )

            labels.append(
                class_name
            )

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        data,
        labels=labels
    )

    plt.title(
        title
    )

    plt.xlabel(
        "Cry Class"
    )

    plt.ylabel(
        ylabel
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        format="svg"
    )

    plt.close()

    print(
        "Created:",
        filename
    )


# ============================================================
# PITCH
# ============================================================

comparison_graph(
    "pitch_mean",
    "Pitch Distribution Across Cry Classes",
    "Mean Pitch (Hz)",
    "02_pitch_comparison.svg"
)


# ============================================================
# RMS
# ============================================================

comparison_graph(
    "rms_mean",
    "RMS Energy Across Cry Classes",
    "Mean RMS Energy",
    "03_rms_comparison.svg"
)


# ============================================================
# STE
# ============================================================

comparison_graph(
    "ste_mean",
    "Short-Time Energy Across Cry Classes",
    "Mean STE",
    "04_ste_comparison.svg"
)


# ============================================================
# ZCR
# ============================================================

comparison_graph(
    "zcr_mean",
    "Zero-Crossing Rate Across Cry Classes",
    "Mean ZCR",
    "05_zcr_comparison.svg"
)


# ============================================================
# MFCC COMPARISON
# ============================================================

print()
print("-" * 70)
print("MFCC ANALYSIS")
print("-" * 70)

mfcc_columns = [
    col for col in df.columns
    if col.startswith("mfcc_")
    and col.endswith("_mean")
]

if len(mfcc_columns) > 0:

    mfcc_means = (
        df.groupby("class")[mfcc_columns]
        .mean()
    )

    plt.figure(
        figsize=(13, 7)
    )

    for class_name in mfcc_means.index:

        plt.plot(
            range(
                1,
                len(mfcc_columns) + 1
            ),
            mfcc_means.loc[class_name].values,
            marker="o",
            label=class_name
        )

    plt.title(
        "Mean MFCC Profile Across Cry Classes"
    )

    plt.xlabel(
        "MFCC Coefficient"
    )

    plt.ylabel(
        "Mean MFCC Value"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "06_mfcc_comparison.svg"
        ),
        format="svg"
    )

    plt.close()

    print(
        "Created: 06_mfcc_comparison.svg"
    )


# ============================================================
# GFCC COMPARISON
# ============================================================

print()
print("-" * 70)
print("GFCC ANALYSIS")
print("-" * 70)

gfcc_columns = [
    col for col in df.columns
    if col.startswith("gfcc_")
    and col.endswith("_mean")
]

if len(gfcc_columns) > 0:

    gfcc_means = (
        df.groupby("class")[gfcc_columns]
        .mean()
    )

    plt.figure(
        figsize=(13, 7)
    )

    for class_name in gfcc_means.index:

        plt.plot(
            range(
                1,
                len(gfcc_columns) + 1
            ),
            gfcc_means.loc[class_name].values,
            marker="o",
            label=class_name
        )

    plt.title(
        "Mean GFCC Profile Across Cry Classes"
    )

    plt.xlabel(
        "GFCC Coefficient"
    )

    plt.ylabel(
        "Mean GFCC Value"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "07_gfcc_comparison.svg"
        ),
        format="svg"
    )

    plt.close()

    print(
        "Created: 07_gfcc_comparison.svg"
    )


# ============================================================
# FEATURE CORRELATION
# ============================================================

print()
print("-" * 70)
print("FEATURE CORRELATION")
print("-" * 70)

feature_columns = [
    col for col in df.columns
    if col not in [
        "filename",
        "class"
    ]
]

numeric_df = df[
    feature_columns
].select_dtypes(
    include="number"
)

if len(numeric_df.columns) > 1:

    correlation = numeric_df.corr()

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.title(
        "Feature Correlation Matrix"
    )

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90,
        fontsize=6
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns,
        fontsize=6
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "08_feature_correlation.svg"
        ),
        format="svg"
    )

    plt.close()

    print(
        "Created: 08_feature_correlation.svg"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Dataset:")
print("  Samples :", len(df))
print("  Columns :", len(df.columns))

print()
print("Classes:")

for class_name, count in class_counts.items():

    print(
        f"  {class_name:<15} : {count}"
    )

print()
print("Missing values:", total_missing)
print("Duplicate rows:", duplicates)

print()
print("Graphs saved in:")
print(
    OUTPUT_DIR
)

print()
print("Generated graphs:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    if filename.endswith(".svg"):

        print(
            "  ",
            filename
        )

print()
print("=" * 70)