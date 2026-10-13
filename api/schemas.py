"""Request and response models for the prediction API.

Validation rules come from the UCI attribute documentation and the EDA:
categorical fields only accept their documented codes, and numeric fields
have generous but physiologically plausible ranges. Anything else is
rejected with HTTP 422 before it reaches the model.

`ca` and `thal` are optional (null allowed) because they were missing for
some patients in the original data; the model pipeline imputes them.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


class PatientFeatures(BaseModel):
    """The 13 clinical attributes of one patient."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLE_PATIENT]},
    )

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical angina, 2 atypical angina, "
                         "3 non-anginal pain, 4 asymptomatic")
    trestbps: float = Field(..., ge=60, le=250,
                            description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700,
                        description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(
        ..., description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: Literal[0, 1, 2] = Field(
        ..., description="Resting ECG: 0 normal, 1 ST-T abnormality, "
                         "2 left ventricular hypertrophy")
    thalach: float = Field(..., ge=50, le=250,
                           description="Maximum heart rate achieved (bpm)")
    exang: Literal[0, 1] = Field(
        ..., description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10,
                           description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(
        ..., description="Peak exercise ST slope: 1 up, 2 flat, 3 down")
    ca: Optional[Literal[0, 1, 2, 3]] = Field(
        None, description="Major vessels coloured by fluoroscopy (0-3); "
                          "null if unknown")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="Thallium test: 3 normal, 6 fixed defect, "
                          "7 reversible defect; null if unknown")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    prediction: Literal[0, 1] = Field(
        ..., description="0 = no disease, 1 = disease")
    label: Literal["no_disease", "disease"]
    disease_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(
        ..., ge=0.5, le=1, description="Probability of the predicted class")
    risk_level: Literal["low", "moderate", "high"] = Field(
        ..., description="low < 0.30 <= moderate < 0.60 <= high")


class PredictionResponse(Prediction):
    model_name: str
    model_version: Optional[int] = None


class BatchResponse(BaseModel):
    model_name: str
    model_version: Optional[int] = None
    n: int
    results: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: Optional[str] = None
    uptime_seconds: float


class ModelInfo(BaseModel):
    model_name: str
    model_label: Optional[str] = None
    registered_version: Optional[int] = None
    threshold: float
    features: list[str]
    metrics: dict
    best_params: dict
    trained_at_utc: Optional[str] = None
    dataset_sha256: Optional[str] = None
    git_commit: Optional[str] = None
