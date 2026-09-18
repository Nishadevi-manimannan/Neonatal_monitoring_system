import os
import joblib
import librosa
import numpy as np
import pandas as pd

# ================================================================
# NEONATAL CRY CLASSIFICATION
# NEW AUDIO PREDICTION
# ================================================================

MODEL_FILE = "Model_Output/neonatal_cry_model.joblib"
FEATURE_FILE = "Model_Output/feature_information.joblib"

print("=" * 70)
print("NEONATAL CRY CLASSIFICATION")
print("NEW AUDIO PREDICTION")
print("=" * 70)

# ================================================================
# CHECK FILES
# ================================================================

if not os.path.exists(MODEL_FILE):
    print("\nERROR: Model not found:")
    print(MODEL_FILE)
    exit()

if not os.path.exists(FEATURE_FILE):
    print("\nERROR: Feature information not found:")
    print(FEATURE_FILE)
    exit()

# ================================================================
# AUDIO FILE
# ================================================================

audio_file = input("\nEnter WAV file path: ").strip().strip('"')

if not os.path.isfile(audio_file):
    print("\nERROR: Audio file not found.")
    exit()

# ================================================================
# LOAD MODEL
# ================================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")

# ================================================================
# LOAD FEATURE INFORMATION
# ================================================================

feature_information = joblib.load(FEATURE_FILE)

# ================================================================
# LOAD AUDIO
# ================================================================

print("\nLoading audio...")

try:

    y, sr = librosa.load(
        audio_file,
        sr=16000,
        mono=True
    )

except Exception as e:

    print("\nERROR while loading audio:")
    print(e)
    exit()

if len(y) == 0:

    print("\nERROR: Audio contains no usable samples.")
    exit()

duration = len(y) / sr

print("Sample rate :", sr)
print("Duration    : %.2f seconds" % duration)

# ================================================================
# PREPROCESSING
# ================================================================

print("\nPreprocessing audio...")

# Remove DC offset
y = y - np.mean(y)

# Normalize amplitude
max_amplitude = np.max(np.abs(y))

if max_amplitude > 0:
    y = y / max_amplitude

print("Preprocessing complete.")

# ================================================================
# FEATURE EXTRACTION
# ================================================================

print("\nExtracting features...")

features = {}

# ================================================================
# 1. PITCH
# ================================================================

f0 = librosa.yin(
    y,
    fmin=80,
    fmax=1000,
    sr=sr
)

f0 = f0[np.isfinite(f0)]

if len(f0) > 0:

    features["pitch_mean"] = np.mean(f0)
    features["pitch_std"] = np.std(f0)
    features["pitch_min"] = np.min(f0)
    features["pitch_max"] = np.max(f0)

else:

    features["pitch_mean"] = 0.0
    features["pitch_std"] = 0.0
    features["pitch_min"] = 0.0
    features["pitch_max"] = 0.0

# ================================================================
# 2. RMS
# ================================================================

rms = librosa.feature.rms(
    y=y
)[0]

features["rms_mean"] = np.mean(rms)
features["rms_std"] = np.std(rms)
features["rms_max"] = np.max(rms)

# ================================================================
# 3. SHORT-TIME ENERGY
# ================================================================

ste = rms ** 2

features["ste_mean"] = np.mean(ste)
features["ste_std"] = np.std(ste)
features["ste_max"] = np.max(ste)

# ================================================================
# 4. ZERO CROSSING RATE
# ================================================================

zcr = librosa.feature.zero_crossing_rate(
    y
)[0]

features["zcr_mean"] = np.mean(zcr)
features["zcr_std"] = np.std(zcr)

# ================================================================
# 5. MFCC
# ================================================================

mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=13
)

for i in range(13):

    number = i + 1

    features[f"mfcc_{number}_mean"] = np.mean(mfcc[i])
    features[f"mfcc_{number}_std"] = np.std(mfcc[i])

# ================================================================
# 6. GFCC
# ================================================================

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=40
)

log_mel = librosa.power_to_db(
    mel,
    ref=np.max
)

gfcc = librosa.feature.mfcc(
    S=log_mel,
    n_mfcc=13
)

for i in range(13):

    number = i + 1

    features[f"gfcc_{number}_mean"] = np.mean(gfcc[i])
    features[f"gfcc_{number}_std"] = np.std(gfcc[i])

# ================================================================
# FEATURE EXTRACTION COMPLETE
# ================================================================

print("\nFeature extraction complete.")
print("Total extracted features:", len(features))

# ================================================================
# EXPECTED 64 FEATURES
# ================================================================

feature_names = [
    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",

    "rms_mean",
    "rms_std",
    "rms_max",

    "ste_mean",
    "ste_std",
    "ste_max",

    "zcr_mean",
    "zcr_std"
]

# MFCC
for i in range(1, 14):
    feature_names.append(f"mfcc_{i}_mean")

for i in range(1, 14):
    feature_names.append(f"mfcc_{i}_std")

# GFCC
for i in range(1, 14):
    feature_names.append(f"gfcc_{i}_mean")

for i in range(1, 14):
    feature_names.append(f"gfcc_{i}_std")

print("\nExpected feature count:", len(feature_names))

# ================================================================
# CHECK FEATURES
# ================================================================

missing = []

for name in feature_names:

    if name not in features:
        missing.append(name)

if len(missing) > 0:

    print("\nERROR: Missing features:")

    for name in missing:
        print("-", name)

    exit()

# ================================================================
# CREATE DATAFRAME
# ================================================================
#
# IMPORTANT:
# The saved model was trained with 64 features.
#
# Therefore we pass all 64 features.
#
# ================================================================

X = pd.DataFrame(
    [[features[name] for name in feature_names]],
    columns=feature_names
)

# Replace invalid values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

print("\nModel input shape:", X.shape)

# ================================================================
# PREDICTION
# ================================================================

print("\nRunning trained Logistic Regression model...")

try:

    prediction = model.predict(X)

except Exception as e:

    print("\nERROR during prediction:")
    print(e)
    exit()

predicted_class = prediction[0]

# ================================================================
# RESULT
# ================================================================

print("\n")
print("=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print("\nAudio file:")
print(os.path.basename(audio_file))

print("\nPredicted cry class:")
print(">>>", predicted_class)

# ================================================================
# PROBABILITIES
# ================================================================

if hasattr(model, "predict_proba"):

    try:

        probabilities = model.predict_proba(X)[0]

        classes = model.classes_

        print("\nPrediction probabilities:")
        print("-" * 45)

        for class_name, probability in zip(
            classes,
            probabilities
        ):

            print(
                "{:<15} : {:6.2f}%".format(
                    class_name,
                    probability * 100
                )
            )

        confidence = np.max(probabilities)

        print("-" * 45)

        print(
            "Confidence       : {:6.2f}%".format(
                confidence * 100
            )
        )

    except Exception as e:

        print("\nCould not calculate probabilities.")
        print(e)

# ================================================================
# END
# ================================================================

print("\n")
print("=" * 70)
print("NEW AUDIO TEST COMPLETE")
print("=" * 70)