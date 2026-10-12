"""Central configuration: paths, column schema and category labels.

Keeping the schema in one place means the download script, cleaning code,
EDA, training, API and tests all agree on column names and types.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
RAW_MANIFEST = RAW_DIR / "manifest.json"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "model.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"

# ---------------------------------------------------------------- source
# Official UCI Machine Learning Repository, Heart Disease (id 45),
# Cleveland Clinic Foundation subset - the 303-patient file used in
# almost every published study on this dataset.
UCI_DATASET_ID = 45
UCI_RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

RANDOM_SEED = 42

# ---------------------------------------------------------------- schema
# The raw file has no header row; these are the 14 attributes in order.
RAW_COLUMNS = [
    "age",       # years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure, mm Hg
    "chol",      # serum cholesterol, mg/dl
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",   # resting ECG result (0-2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise-induced angina (1 = yes)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # slope of peak exercise ST segment (1-3)
    "ca",        # number of major vessels coloured by fluoroscopy (0-3)
    "thal",      # thallium stress test result (3, 6, 7)
    "num",       # diagnosis: 0 = no disease, 1-4 = increasing severity
]

TARGET = "target"  # binary target derived from `num`

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Valid codes per categorical column (used for validation and tests).
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}

# Human-readable labels, used only for plots and reporting.
CATEGORY_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina",
           3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
    TARGET: {0: "No disease", 1: "Disease"},
}

FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved (bpm)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Peak exercise ST slope",
    "ca": "Major vessels coloured (ca)",
    "thal": "Thallium stress test",
    TARGET: "Heart disease",
}
