"""FastAPI REST service providing real-time risk scoring, resource allocation, and decision logging."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure src/ is on sys.path
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from resq.allocation import allocate_units_to_incidents
from resq.config import DEFAULT_CITY, DEFAULT_MODELS_DIR
from resq.data_prep import FEATURE_COLUMNS
from resq.explain import ResQExplainer
from resq.ibmz import get_platform_telemetry
from resq.models import ResQRiskModel, train_and_compare_models
from resq.ranking import rank_incidents_ai
from resq.simulator import EmergencySimulator, Incident
from resq.storage import storage
from resq.stream import stream_manager

logger = logging.getLogger(__name__)

# Global model and explainer instances
champion_model: Optional[ResQRiskModel] = None
explainer: Optional[ResQExplainer] = None
active_simulator: Optional[EmergencySimulator] = None


def ensure_model_loaded() -> ResQRiskModel:
    """Ensure a trained risk model is loaded; train a fast seed model if none exists."""
    global champion_model, explainer, active_simulator
    if champion_model is not None:
        return champion_model

    model_path = os.path.join(DEFAULT_MODELS_DIR, "champion_model.joblib")
    if os.path.exists(model_path):
        try:
            champion_model = ResQRiskModel.load(model_path)
            explainer = ResQExplainer(champion_model)
            logger.info("Loaded persisted model: %s", champion_model.model_name)
            return champion_model
        except Exception as exc:
            logger.warning("Could not load model (%s); training initial seed model.", exc)

    # Train initial model using simulated baseline
    logger.info("Generating seed dataset and training initial model...")
    active_simulator = EmergencySimulator(seed=42)
    df_seed = active_simulator.generate_batch_dataset(num_ticks=40)
    champion_model, _ = train_and_compare_models(df_seed, seed=42)
    champion_model.save(DEFAULT_MODELS_DIR)
    explainer = ResQExplainer(champion_model)
    return champion_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup initialization and teardown hook."""
    global active_simulator
    ensure_model_loaded()
    if active_simulator is None:
        active_simulator = EmergencySimulator(seed=42)
    yield


app = FastAPI(
    title="ResQ-AI Emergency Decision-Support API",
    description="Real-Time AI Decision-Support for Emergency Incident Prioritization & Resource Allocation (IBM Z Datathon 2026). Human-in-the-Loop advisory system.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request & Response Schemas
class IncidentInputSchema(BaseModel):
    incident_id: str = Field(default="INC-NEW-01", description="Unique incident identifier")
    incident_type: str = Field(default="road_accident", description="road_accident, fire, flood, medical_emergency, building_collapse")
    severity: int = Field(default=4, ge=1, le=5, description="Incident severity level (1-5)")
    latitude: float = Field(default=28.6139, description="Incident latitude")
    longitude: float = Field(default=77.2090, description="Incident longitude")
    people_affected: int = Field(default=5, ge=0, description="Estimated casualties / people affected")
    road_type: str = Field(default="highway", description="highway, arterial, local, bridge")
    traffic_congestion: float = Field(default=0.75, ge=0.0, le=1.0, description="Congestion index (0.0 to 1.0)")
    temperature_c: float = Field(default=30.0, description="Temperature in Celsius")
    rain_mm: float = Field(default=15.0, ge=0.0, description="Rainfall in mm/h")
    wind_kmh: float = Field(default=22.0, ge=0.0, description="Wind speed in km/h")
    visibility_km: float = Field(default=3.5, ge=0.1, description="Visibility in km")


class PredictResponseSchema(BaseModel):
    incident_id: str
    risk_score: float
    is_high_risk: bool
    urgency_score: float
    plain_language_reason: str
    top_attributions: List[Dict[str, Any]]
    model_used: str
    disclaimer: str = "Decision support only. Human approval required prior to any dispatch action."


class DecisionRequestSchema(BaseModel):
    incident_id: str
    commander_id: str = Field(default="COMM-01", description="Human dispatcher / incident commander ID")
    decision: str = Field(default="APPROVE", description="APPROVE, REJECT, or OVERRIDE")
    assigned_unit_id: Optional[str] = Field(default=None, description="Assigned unit ID if approved")
    comments: str = Field(default="", description="Operational notes or rationale")


class DecisionResponseSchema(BaseModel):
    decision_id: int
    incident_id: str
    commander_id: str
    decision: str
    assigned_unit_id: Optional[str]
    status: str
    timestamp: str


@app.get("/health", tags=["System"])
def get_health() -> Dict[str, Any]:
    """Health check validating database, streaming fallback, and model status."""
    model_status = "READY" if champion_model is not None else "UNINITIALIZED"
    stream_status = stream_manager.get_status()
    return {
        "status": "HEALTHY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_status": model_status,
        "champion_model": champion_model.model_name if champion_model else None,
        "streaming": stream_status,
        "human_in_the_loop_active": True,
    }


@app.get("/platform", tags=["System"])
def get_platform() -> Dict[str, Any]:
    """Inspect host architecture, Python version, and IBM Z / Snap ML accelerator telemetry."""
    telemetry = get_platform_telemetry()
    return telemetry.__dict__


@app.post("/predict", response_model=PredictResponseSchema, tags=["Scoring"])
def predict_incident_risk(incident: IncidentInputSchema) -> PredictResponseSchema:
    """Score a single incident, calculate urgency, and return SHAP-based plain English explanations."""
    model = ensure_model_loaded()
    inc_dict = incident.model_dump()
    inc_df = pd.DataFrame([inc_dict])

    probs = model.predict_risk_proba(inc_df)
    risk_score = float(probs[0])
    is_high = bool(risk_score >= 0.55)

    # Compute urgency score
    ranked_df = rank_incidents_ai(inc_df, risk_scores=probs)
    urgency = float(ranked_df["urgency_score"].iloc[0])

    # Explain using SHAP
    top_attrs = []
    plain_reason = "Standard emergency profile based on reported attributes."
    if explainer is not None:
        attrs, plain_reason = explainer.explain_incident(inc_df.iloc[0])
        top_attrs = [a.__dict__ for a in attrs]

    # Save to SQLite and Stream
    rec_save = {**inc_dict, "risk_score": risk_score, "urgency_score": urgency}
    storage.save_incidents([rec_save])
    stream_manager.publish_event("resq:incidents", rec_save)

    return PredictResponseSchema(
        incident_id=incident.incident_id,
        risk_score=round(risk_score, 4),
        is_high_risk=is_high,
        urgency_score=round(urgency, 4),
        plain_language_reason=plain_reason,
        top_attributions=top_attrs,
        model_used=model.model_name,
    )


@app.get("/incidents", tags=["Operations"])
def get_ranked_incidents(
    status: Optional[str] = Query(default=None, description="Optional filter by status (PENDING, APPROVED, REJECTED)"),
    limit: int = Query(default=50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """Retrieve ranked incident queue from database."""
    incidents = storage.get_incidents(status=status, limit=limit)
    return incidents


@app.get("/recommendations", tags=["Operations"])
def get_recommendations(
    incident_id: Optional[str] = Query(default=None, description="Optional filter by incident ID"),
) -> List[Dict[str, Any]]:
    """Retrieve current AI-recommended unit dispatches awaiting human approval."""
    recs = storage.get_recommendations(incident_id=incident_id)
    return recs


@app.post("/decisions", response_model=DecisionResponseSchema, tags=["Operations"])
def log_decision(payload: DecisionRequestSchema) -> DecisionResponseSchema:
    """Record human incident commander approval or rejection for an advisory recommendation."""
    decision_id = storage.log_human_decision(
        incident_id=payload.incident_id,
        commander_id=payload.commander_id,
        decision=payload.decision.upper(),
        assigned_unit_id=payload.assigned_unit_id,
        comments=payload.comments,
    )

    # Publish audit event
    stream_manager.publish_event("resq:decisions", {
        "decision_id": decision_id,
        "incident_id": payload.incident_id,
        "commander_id": payload.commander_id,
        "decision": payload.decision.upper(),
        "assigned_unit_id": payload.assigned_unit_id,
    })

    return DecisionResponseSchema(
        decision_id=decision_id,
        incident_id=payload.incident_id,
        commander_id=payload.commander_id,
        decision=payload.decision.upper(),
        assigned_unit_id=payload.assigned_unit_id,
        status="RECORDED",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
