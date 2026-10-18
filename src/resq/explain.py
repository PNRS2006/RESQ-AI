"""SHAP explainability engine producing local feature attributions and plain-English dispatch rationale."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from resq.data_prep import FEATURE_COLUMNS, prepare_feature_matrix
from resq.models import ResQRiskModel

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class FeatureAttribution:
    """Individual feature contribution to the predicted risk score."""
    feature_name: str
    feature_value: float
    attribution_value: float
    direction: str  # 'increases_risk' or 'decreases_risk'


# Human-friendly descriptions for individual features
FEATURE_NARRATIVES: Dict[str, Tuple[str, str]] = {
    "severity": ("High reported severity (Level {val})", "Low reported severity (Level {val})"),
    "people_affected": ("High casualty estimate ({val:.0f} people affected)", "Minimal casualties ({val:.0f} people affected)"),
    "traffic_congestion": ("Gridlock traffic congestion ({val:.2f})", "Smooth traffic flow ({val:.2f})"),
    "rain_mm": ("Heavy precipitation ({val:.1f} mm/h)", "Dry weather conditions ({val:.1f} mm/h)"),
    "wind_kmh": ("High wind speeds ({val:.1f} km/h)", "Calm winds ({val:.1f} km/h)"),
    "visibility_hazard": ("Severe visibility impairment ({val:.1f} km deficit)", "Clear atmospheric visibility"),
    "rain_x_congestion": ("Compounding rain and heavy traffic gridlock", "Favorable road conditions"),
    "is_highway_bridge": ("Location on high-speed highway or bridge structure", "Local street corridor"),
    "is_night_or_rush": ("Peak rush hour / night-time response window", "Normal daytime operational window"),
    "type_building_collapse": ("Structural building collapse hazard", "Non-structural incident"),
    "type_fire": ("Active fire outbreak hazard", "Non-fire incident"),
    "type_flood": ("Submerged roadways and flash flooding hazard", "Non-flood incident"),
    "type_road_accident": ("Road collision traffic incident", "Non-vehicular incident"),
    "type_medical_emergency": ("Acute medical emergency response", "Non-medical incident"),
}


class ResQExplainer:
    """Explains predictions using TreeSHAP with intelligent fallback heuristics."""

    def __init__(self, risk_model: ResQRiskModel, background_df: Optional[pd.DataFrame] = None):
        self.risk_model = risk_model
        self.shap_explainer: Optional[Any] = None

        if HAS_SHAP and hasattr(risk_model.estimator, "predict_proba"):
            try:
                # TreeExplainer for tree ensembles (RandomForest, XGBoost, LightGBM)
                raw_model = risk_model.estimator
                if hasattr(raw_model, "named_steps"):
                    raw_model = raw_model.named_steps.get("clf", raw_model)
                self.shap_explainer = shap.TreeExplainer(raw_model)
                logger.info("Initialized TreeSHAP explainer for %s", risk_model.model_name)
            except Exception as exc:
                logger.debug("TreeExplainer init failed (%s); using feature attribution fallback.", exc)
                self.shap_explainer = None

    def explain_incident(
        self,
        incident_row: pd.Series,
        top_k: int = 4,
    ) -> Tuple[List[FeatureAttribution], str]:
        """Compute top local feature attributions and translate into plain English."""
        df_single = pd.DataFrame([incident_row])
        X = prepare_feature_matrix(df_single)

        attributions: List[FeatureAttribution] = []

        if self.shap_explainer is not None:
            try:
                raw_shap = self.shap_explainer.shap_values(X)
                # Handle binary classification output format differences in SHAP versions
                if isinstance(raw_shap, list) and len(raw_shap) == 2:
                    vals = np.array(raw_shap[1])[0]
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                    vals = raw_shap[0, :, 1]
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
                    vals = raw_shap[0]
                else:
                    vals = np.zeros(len(FEATURE_COLUMNS))

                for col, s_val in zip(FEATURE_COLUMNS, vals):
                    f_val = float(X[col].iloc[0])
                    attributions.append(
                        FeatureAttribution(
                            feature_name=col,
                            feature_value=round(f_val, 2),
                            attribution_value=round(float(s_val), 4),
                            direction="increases_risk" if s_val > 0 else "decreases_risk",
                        )
                    )
            except Exception as exc:
                logger.debug("SHAP explanation call failed (%s); falling back to heuristic attribution.", exc)
                attributions = []

        if not attributions:
            # Fallback based on normalized feature weights
            severity = float(incident_row.get("severity", 3))
            people = float(incident_row.get("people_affected", 0))
            congestion = float(incident_row.get("traffic_congestion", 0.5))
            rain = float(incident_row.get("rain_mm", 0.0))
            visibility = float(incident_row.get("visibility_km", 10.0))

            heuristics = [
                ("severity", severity, (severity - 3.0) * 0.15),
                ("people_affected", people, min(people / 20.0, 1.0) * 0.20),
                ("traffic_congestion", congestion, (congestion - 0.5) * 0.12),
                ("rain_mm", rain, min(rain / 30.0, 1.0) * 0.10),
                ("visibility_hazard", max(0.0, 10.0 - visibility), max(0.0, 10.0 - visibility) * 0.015),
            ]
            for f_name, f_val, attr_val in heuristics:
                attributions.append(
                    FeatureAttribution(
                        feature_name=f_name,
                        feature_value=round(f_val, 2),
                        attribution_value=round(attr_val, 4),
                        direction="increases_risk" if attr_val > 0 else "decreases_risk",
                    )
                )

        # Sort by absolute attribution magnitude
        attributions.sort(key=lambda a: abs(a.attribution_value), reverse=True)
        top_attributions = attributions[:top_k]

        # Generate human-readable narrative
        plain_reasons = []
        for attr in top_attributions:
            narratives = FEATURE_NARRATIVES.get(attr.feature_name)
            if narratives:
                template = narratives[0] if attr.attribution_value >= 0 else narratives[1]
                try:
                    reason = template.format(val=attr.feature_value)
                except Exception:
                    reason = template
                plain_reasons.append(reason)

        if plain_reasons:
            plain_summary = "; ".join(plain_reasons[:3]) + "."
        else:
            plain_summary = "Standard emergency profile based on reported location and baseline severity."

        return top_attributions, plain_summary
