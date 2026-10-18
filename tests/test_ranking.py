"""Unit tests for multi-factor urgency ranking and rule-based CAD baseline."""

import numpy as np
import pandas as pd
from resq.ranking import compute_rule_baseline_ranking, compute_urgency_scores, rank_incidents_ai


def test_compute_urgency_scores():
    """Verify urgency score properly combines risk, casualties, and elapsed wait time."""
    df = pd.DataFrame([
        {"risk_score": 0.90, "people_affected": 20, "timestamp": "2026-07-15 14:00:00"},
        {"risk_score": 0.20, "people_affected": 0, "timestamp": "2026-07-15 14:00:00"},
    ])

    scores = compute_urgency_scores(df)
    assert len(scores) == 2
    assert scores[0] > scores[1]
    assert 0.0 <= scores[0] <= 1.0


def test_rank_incidents_ai():
    """Verify AI ranking sorts incidents descending by urgency score."""
    df = pd.DataFrame([
        {"incident_id": "INC-LOW", "severity": 2, "people_affected": 1, "latitude": 28.6, "longitude": 77.2},
        {"incident_id": "INC-CRIT", "severity": 5, "people_affected": 15, "latitude": 28.7, "longitude": 77.3},
    ])
    risks = np.array([0.25, 0.92])
    ranked = rank_incidents_ai(df, risk_scores=risks)

    assert ranked["incident_id"].iloc[0] == "INC-CRIT"
    assert ranked["ai_rank"].iloc[0] == 1
    assert ranked["incident_id"].iloc[1] == "INC-LOW"
    assert ranked["ai_rank"].iloc[1] == 2


def test_compute_rule_baseline_ranking():
    """Verify deterministic rule baseline orders by severity then casualties."""
    df = pd.DataFrame([
        {"incident_id": "INC-A", "severity": 3, "people_affected": 10},
        {"incident_id": "INC-B", "severity": 5, "people_affected": 2},
        {"incident_id": "INC-C", "severity": 5, "people_affected": 8},
    ])
    ranked = compute_rule_baseline_ranking(df)

    assert ranked["incident_id"].iloc[0] == "INC-C"  # Severity 5, 8 casualties
    assert ranked["incident_id"].iloc[1] == "INC-B"  # Severity 5, 2 casualties
    assert ranked["incident_id"].iloc[2] == "INC-A"  # Severity 3
