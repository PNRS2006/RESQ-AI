"""Multi-model training, model selection, evaluation, and risk scoring engine."""

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from resq.config import DEFAULT_MODELS_DIR, RANDOM_SEED
from resq.data_prep import FEATURE_COLUMNS, prepare_feature_matrix, temporal_train_test_split

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationMetrics:
    """Standard evaluation metrics for binary risk classification."""
    model_name: str
    roc_auc: float
    pr_auc: float
    f1_score: float
    accuracy: float
    brier_score: float


class ResQRiskModel:
    """Wrapper holding trained model, scaler/pipeline, feature names, and metadata."""

    def __init__(
        self,
        model_name: str,
        estimator: Any,
        features: List[str] = FEATURE_COLUMNS,
        metrics: Optional[ModelEvaluationMetrics] = None,
    ):
        self.model_name = model_name
        self.estimator = estimator
        self.features = features
        self.metrics = metrics

    def predict_risk_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict risk probability (0.0 to 1.0) for an input dataframe."""
        X = prepare_feature_matrix(df)
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X)[:, 1]
        elif hasattr(self.estimator, "decision_function"):
            raw = self.estimator.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-raw))
        else:
            probs = self.estimator.predict(X)
        return np.clip(probs, 0.01, 0.99)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Standard 2D probability array format [P(0), P(1)]."""
        p1 = self.predict_risk_proba(df)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def save(self, output_dir: str = DEFAULT_MODELS_DIR) -> str:
        """Persist model and metadata to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_file = out_path / "champion_model.joblib"
        meta_file = out_path / "champion_metrics.json"

        joblib.dump(self, model_file)

        if self.metrics:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.metrics), f, indent=2)

        logger.info("Saved champion model %s to %s", self.model_name, model_file)
        return str(model_file)

    @classmethod
    def load(cls, model_path: str = f"{DEFAULT_MODELS_DIR}/champion_model.joblib") -> "ResQRiskModel":
        """Load persisted model artifact."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        loaded = joblib.load(model_path)
        return loaded


def train_and_compare_models(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    seed: int = RANDOM_SEED,
) -> Tuple[ResQRiskModel, List[ModelEvaluationMetrics]]:
    """Train Logistic Regression, Random Forest, XGBoost, and LightGBM.
    
    Selects the best model based on held-out validation ROC-AUC / PR-AUC.
    """
    train_df, test_df = temporal_train_test_split(df, test_ratio=test_ratio)

    X_train = prepare_feature_matrix(train_df)
    y_train = train_df["is_high_risk"].values

    X_test = prepare_feature_matrix(test_df)
    y_test = test_df["is_high_risk"].values

    candidates: Dict[str, Any] = {}

    # 1. Baseline Logistic Regression (with StandardScaler)
    candidates["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
    ])

    # 2. Random Forest
    candidates["RandomForest"] = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )

    # 3. XGBoost
    if HAS_XGBOOST:
        candidates["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=seed,
            eval_metric="logloss",
        )

    # 4. LightGBM
    if HAS_LIGHTGBM:
        candidates["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=seed,
            verbose=-1,
        )

    all_metrics: List[ModelEvaluationMetrics] = []
    best_score = -1.0
    best_candidate_name = ""
    fitted_models: Dict[str, Any] = {}

    for name, model in candidates.items():
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Evaluate on validation split
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            raw = model.decision_function(X_test)
            probs = 1.0 / (1.0 + np.exp(-raw))

        preds = (probs >= 0.50).astype(int)

        roc = float(roc_auc_score(y_test, probs))
        pr = float(average_precision_score(y_test, probs))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        acc = float(accuracy_score(y_test, preds))
        brier = float(brier_score_loss(y_test, probs))

        metric = ModelEvaluationMetrics(
            model_name=name,
            roc_auc=round(roc, 4),
            pr_auc=round(pr, 4),
            f1_score=round(f1, 4),
            accuracy=round(acc, 4),
            brier_score=round(brier, 4),
        )
        all_metrics.append(metric)

        # Champion criterion: highest PR-AUC + ROC-AUC composite
        composite_score = (roc + pr) / 2.0
        if composite_score > best_score:
            best_score = composite_score
            best_candidate_name = name

    logger.info("Selected Champion Model: %s (Composite: %.4f)", best_candidate_name, best_score)

    champion_metric = next(m for m in all_metrics if m.model_name == best_candidate_name)
    champion_model = ResQRiskModel(
        model_name=best_candidate_name,
        estimator=fitted_models[best_candidate_name],
        features=FEATURE_COLUMNS,
        metrics=champion_metric,
    )

    return champion_model, all_metrics
