"""Urgency calculation, incident ranking, and rule-based baseline comparator."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from resq.config import URGENCY_WEIGHTS


def compute_urgency_scores(
    df: pd.DataFrame,
    risk_scores: Optional[np.ndarray] = None,
    current_time: Optional[datetime] = None,
    weights: Dict[str, float] = URGENCY_WEIGHTS,
) -> np.ndarray:
    """Compute human-centered urgency scores combining AI risk, casualties, and elapsed wait time.
    
    Formula:
      Urgency = w_risk * risk_score + w_cas * norm_casualties + w_time * norm_time_elapsed
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)

    # 1. AI Risk score component (0 to 1)
    if risk_scores is not None and len(risk_scores) == n:
        risk_comp = np.clip(np.array(risk_scores, dtype=float), 0.0, 1.0)
    elif "risk_score" in df.columns:
        risk_comp = np.clip(df["risk_score"].fillna(0.5).values.astype(float), 0.0, 1.0)
    elif "true_risk_score" in df.columns:
        risk_comp = np.clip(df["true_risk_score"].fillna(0.5).values.astype(float), 0.0, 1.0)
    else:
        risk_comp = np.full(n, 0.5)

    # 2. Casualty count component (normalized by soft ceiling of 25 people)
    if "people_affected" in df.columns:
        peop = df["people_affected"].fillna(0).values.astype(float)
        cas_comp = np.clip(peop / 25.0, 0.0, 1.0)
    else:
        cas_comp = np.zeros(n)

    # 3. Elapsed time component (minutes waiting since report, normalized by 60 mins)
    if "timestamp" in df.columns and current_time is not None:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        time_diffs_min = np.maximum(
            0.0,
            [(current_time - ts).total_seconds() / 60.0 if pd.notnull(ts) else 15.0 for ts in timestamps]
        )
        time_comp = np.clip(np.array(time_diffs_min) / 60.0, 0.0, 1.0)
    elif "tick" in df.columns and "current_tick" in df.columns:
        tick_diff = np.maximum(0, df["current_tick"] - df["tick"])
        time_comp = np.clip(tick_diff / 4.0, 0.0, 1.0)
    else:
        # Default mild decay assumption (15 mins elapsed)
        time_comp = np.full(n, 0.25)

    w_risk = weights.get("risk", 0.50)
    w_cas = weights.get("casualties", 0.30)
    w_time = weights.get("time_elapsed", 0.20)

    total_weight = w_risk + w_cas + w_time
    w_risk /= total_weight
    w_cas /= total_weight
    w_time /= total_weight

    urgency = (w_risk * risk_comp) + (w_cas * cas_comp) + (w_time * time_comp)
    return np.round(np.clip(urgency, 0.01, 0.99), 4)


def rank_incidents_ai(
    df: pd.DataFrame,
    risk_scores: Optional[np.ndarray] = None,
    current_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Rank incidents using ResQ-AI multi-factor urgency scoring."""
    df_out = df.copy()
    if risk_scores is not None:
        df_out["risk_score"] = risk_scores

    urgency_scores = compute_urgency_scores(df_out, risk_scores=risk_scores, current_time=current_time)
    df_out["urgency_score"] = urgency_scores

    # Sort descending by urgency score
    df_ranked = df_out.sort_values(
        by=["urgency_score", "people_affected", "severity"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df_ranked["ai_rank"] = np.arange(1, len(df_ranked) + 1)
    return df_ranked


def compute_rule_baseline_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic severity-first baseline ranking for direct comparison.
    
    Standard CAD Rule:
    1. Highest Severity First (5 down to 1)
    2. Highest People Affected
    3. Oldest report first (earliest timestamp)
    """
    df_base = df.copy()
    sort_cols = []
    ascending_flags = []

    if "severity" in df_base.columns:
        sort_cols.append("severity")
        ascending_flags.append(False)
    if "people_affected" in df_base.columns:
        sort_cols.append("people_affected")
        ascending_flags.append(False)
    if "timestamp" in df_base.columns:
        sort_cols.append("timestamp")
        ascending_flags.append(True)

    if sort_cols:
        df_base = df_base.sort_values(by=sort_cols, ascending=ascending_flags).reset_index(drop=True)
    else:
        df_base = df_base.reset_index(drop=True)

    df_base["baseline_rank"] = np.arange(1, len(df_base) + 1)
    return df_base
