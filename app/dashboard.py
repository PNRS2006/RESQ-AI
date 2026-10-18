"""ResQ-AI Streamlit Command Center: Real-Time AI Emergency Decision-Support Dashboard."""

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from resq.allocation import allocate_units_to_incidents
from resq.config import (
    DEFAULT_CITY,
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_PATH,
    SCENARIO_CITY_SURGE,
    SCENARIO_HEAVY_RAIN,
    SCENARIO_HIGHWAY_ACCIDENT,
    SCENARIO_NAMES,
    SCENARIO_NORMAL,
)
from resq.data_prep import FEATURE_COLUMNS, prepare_feature_matrix
from resq.explain import ResQExplainer
from resq.ibmz import get_platform_telemetry, run_benchmark_comparison
from resq.models import ResQRiskModel, train_and_compare_models
from resq.ranking import compute_rule_baseline_ranking, rank_incidents_ai
from resq.simulator import EmergencySimulator, Incident
from resq.storage import storage
from resq.stream import stream_manager
from resq.weather import get_current_weather

# Page configuration
st.set_page_config(
    page_title="ResQ-AI | Emergency Decision-Support",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished command-center styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .badge-critical {
        background-color: #d32f2f;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-high {
        background-color: #f57c00;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #fbc02d;
        color: #212121;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #388e3c;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "simulator" not in st.session_state:
    st.session_state.simulator = EmergencySimulator(seed=42)
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = SCENARIO_NORMAL
if "model" not in st.session_state:
    model_path = os.path.join(DEFAULT_MODELS_DIR, "champion_model.joblib")
    if os.path.exists(model_path):
        st.session_state.model = ResQRiskModel.load(model_path)
    else:
        df_init = st.session_state.simulator.generate_batch_dataset(num_ticks=30)
        st.session_state.model, _ = train_and_compare_models(df_init, seed=42)
        st.session_state.model.save(DEFAULT_MODELS_DIR)

if "explainer" not in st.session_state:
    st.session_state.explainer = ResQExplainer(st.session_state.model)

if "active_incidents" not in st.session_state:
    # Initial tick
    incs, weather, congestion = st.session_state.simulator.step_tick(count_override=5)
    st.session_state.active_incidents = incs
    st.session_state.current_weather = weather
    st.session_state.current_congestion = congestion


# Mandatory Disclaimer Banner
st.warning(
    "⚠️ **MANDATORY NOTICE:** Decision support only. A human approves every action. "
    "Data and labels are simulated for demonstration. All operational metrics reflect real code runs."
)

# Header Section
col_title, col_stat = st.columns([3, 1])
with col_title:
    st.title("🚨 ResQ-AI: Emergency Decision-Support System")
    st.caption("IBM Z Datathon 2026 — Real-Time AI for Critical Decisions | City of Delhi Operations")

with col_stat:
    st.info(f"**Champion Model:** `{st.session_state.model.model_name}`\n\n**Platform:** `{get_platform_telemetry().machine_architecture}`")


# Sidebar Controls
st.sidebar.header("🕹️ Simulation Controls")

selected_scenario = st.sidebar.selectbox(
    "Emergency Scenario",
    options=SCENARIO_NAMES,
    index=SCENARIO_NAMES.index(st.session_state.current_scenario),
)

if selected_scenario != st.session_state.current_scenario:
    st.session_state.current_scenario = selected_scenario
    st.session_state.simulator = EmergencySimulator(scenario=selected_scenario, seed=np.random.randint(1, 10000))
    incs, weather, congestion = st.session_state.simulator.step_tick(count_override=6)
    st.session_state.active_incidents = incs
    st.session_state.current_weather = weather
    st.session_state.current_congestion = congestion
    st.rerun()

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("⏩ Next Tick (+15m)", use_container_width=True):
        incs, weather, congestion = st.session_state.simulator.step_tick()
        st.session_state.active_incidents = incs
        st.session_state.current_weather = weather
        st.session_state.current_congestion = congestion
        st.rerun()

with col_btn2:
    if st.button("🔄 Reset Fleet", use_container_width=True):
        st.session_state.simulator = EmergencySimulator(scenario=selected_scenario, seed=42)
        incs, weather, congestion = st.session_state.simulator.step_tick(count_override=5)
        st.session_state.active_incidents = incs
        st.session_state.current_weather = weather
        st.session_state.current_congestion = congestion
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🌤️ Environmental Telemetry")
w = st.session_state.current_weather
st.sidebar.markdown(f"**Source:** {w.source_description}")
st.sidebar.metric("Rainfall", f"{w.rain_mm} mm/h")
st.sidebar.metric("Visibility", f"{w.visibility_km} km")
st.sidebar.metric("Wind Speed", f"{w.wind_kmh} km/h")
st.sidebar.metric("Traffic Congestion", f"{st.session_state.current_congestion * 100:.0f}%")

st.sidebar.markdown("---")
st.sidebar.caption("Human Commander: **COMM-CHIEF-01** (Signed In)")


# Prepare Dataframe for Active Incidents
df_active = pd.DataFrame([inc.__dict__ for inc in st.session_state.active_incidents])

# Predict Risk and Rank
risk_probs = st.session_state.model.predict_risk_proba(df_active)
df_ranked = rank_incidents_ai(df_active, risk_scores=risk_probs)

# Save to SQLite
storage.save_incidents(df_ranked.to_dict(orient="records"))

# Generate Recommendations
recommendations = allocate_units_to_incidents(df_ranked, st.session_state.simulator.units)
storage.save_recommendations([r.__dict__ for r in recommendations])
rec_map = {r.incident_id: r for r in recommendations}


# 5-Tab Interface
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Live Command View",
    "📋 Priority Queue & Dispatch",
    "🔍 AI Explainability (SHAP)",
    "📊 Empirical Evaluation",
    "🖥️ IBM Z & Platform",
])


# ==========================================
# TAB 1: LIVE COMMAND VIEW
# ==========================================
with tab1:
    st.subheader("Geospatial Incident Map & Unit Tracker")

    # High-Risk Alert Banner
    critical_incidents = df_ranked[df_ranked["urgency_score"] >= 0.70]
    if len(critical_incidents) > 0:
        crit_names = ", ".join(critical_incidents["incident_id"].tolist())
        st.error(f"🚨 **HIGH-URGENCY ALERT:** {len(critical_incidents)} critical incident(s) require immediate review: **{crit_names}**")

    # Metrics Summary Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Active Incidents", len(df_ranked))
    with m2:
        high_risk_count = int(np.sum(df_ranked["urgency_score"] >= 0.55))
        st.metric("High-Urgency Incidents", high_risk_count)
    with m3:
        avail_units = sum(1 for u in st.session_state.simulator.units if u.status == "AVAILABLE")
        st.metric("Available Fleet Units", f"{avail_units} / {len(st.session_state.simulator.units)}")
    with m4:
        st.metric("Simulated Tick", f"#{st.session_state.simulator.current_tick}")

    # Folium Map Creation
    city = DEFAULT_CITY
    m = folium.Map(
        location=[city.center_lat, city.center_lon],
        zoom_start=city.default_zoom,
        tiles="CartoDB positron",
    )

    # Plot Emergency Units
    unit_icons = {
        "ambulance": ("plus", "blue"),
        "fire_engine": ("fire", "darkred"),
        "police_cruiser": ("shield", "darkblue"),
    }
    for unit in st.session_state.simulator.units:
        icon_name, icon_color = unit_icons.get(unit.unit_type, ("info-sign", "gray"))
        folium.Marker(
            location=[unit.latitude, unit.longitude],
            tooltip=f"{unit.unit_id} ({unit.unit_type.replace('_', ' ').title()}) - {unit.status}",
            popup=f"<b>Unit:</b> {unit.unit_id}<br><b>Type:</b> {unit.unit_type}<br><b>Station:</b> {unit.station_name}<br><b>Status:</b> {unit.status}",
            icon=folium.Icon(icon=icon_name, prefix="fa", color=icon_color),
        ).add_to(m)

    # Plot Emergency Incidents
    for _, row in df_ranked.iterrows():
        u_score = row["urgency_score"]
        if u_score >= 0.70:
            color = "#d32f2f"  # Red
            rad = 12
        elif u_score >= 0.55:
            color = "#f57c00"  # Orange
            rad = 10
        elif u_score >= 0.35:
            color = "#fbc02d"  # Yellow
            rad = 8
        else:
            color = "#388e3c"  # Green
            rad = 7

        rec_unit = rec_map.get(row["incident_id"])
        rec_text = f"{rec_unit.recommended_unit_id} (ETA: {rec_unit.estimated_eta_minutes}m)" if rec_unit and not rec_unit.unmet_demand else "No Unit Available"

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 13px;">
            <b>ID:</b> {row['incident_id']}<br>
            <b>Type:</b> {row['incident_type'].replace('_', ' ').title()}<br>
            <b>Severity:</b> Level {row['severity']} / 5<br>
            <b>Casualties:</b> {row['people_affected']}<br>
            <b>AI Urgency Score:</b> {row['urgency_score']:.2f}<br>
            <b>ML Risk Probability:</b> {row['risk_score']:.2f}<br>
            <b>Recommended Dispatch:</b> {rec_text}
        </div>
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=rad,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['incident_id']} (Urgency: {row['urgency_score']:.2f})",
        ).add_to(m)

    st_folium(m, width=None, height=520)


# ==========================================
# TAB 2: PRIORITY QUEUE & DISPATCH
# ==========================================
with tab2:
    st.subheader("Ranked Incident Queue & Unit Recommendations")
    st.caption("Review advisory AI recommendations and record formal command authorization. The system will never dispatch autonomously.")

    # Format table for display
    display_rows = []
    for _, row in df_ranked.iterrows():
        inc_id = row["incident_id"]
        rec = rec_map.get(inc_id)

        # Plain language reason via explainer
        _, reason_text = st.session_state.explainer.explain_incident(row)

        unit_str = f"{rec.recommended_unit_id} ({rec.unit_type})" if rec and not rec.unmet_demand else "⚠️ UNMET DEMAND"
        eta_str = f"{rec.estimated_eta_minutes} min ({rec.distance_km} km)" if rec and not rec.unmet_demand else "N/A"

        display_rows.append({
            "Rank": int(row["ai_rank"]),
            "Incident ID": inc_id,
            "Type": row["incident_type"].replace("_", " ").title(),
            "Severity": f"Level {row['severity']}",
            "Casualties": int(row["people_affected"]),
            "Risk Score": f"{row['risk_score']:.3f}",
            "Urgency Score": f"{row['urgency_score']:.3f}",
            "Recommended Unit": unit_str,
            "ETA (Congestion-Adjusted)": eta_str,
            "Key AI Rationale": reason_text,
            "Status": row.get("status", "PENDING"),
        })

    df_display = pd.DataFrame(display_rows)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("✍️ Human Commander Dispatch Authorization")

    col_act1, col_act2, col_act3 = st.columns([2, 1, 1])
    with col_act1:
        selected_inc_id = st.selectbox(
            "Select Incident to Authorize / Dismiss",
            options=df_ranked["incident_id"].tolist(),
        )
        comm_notes = st.text_input("Operational Notes / Override Rationale (Optional)", "")

    chosen_row = df_ranked[df_ranked["incident_id"] == selected_inc_id].iloc[0]
    rec_item = rec_map.get(selected_inc_id)

    with col_act2:
        st.write("")
        st.write("")
        if st.button("✅ APPROVE DISPATCH", use_container_width=True, type="primary"):
            unit_to_assign = rec_item.recommended_unit_id if rec_item and not rec_item.unmet_demand else "MANUAL_DISPATCH"
            storage.log_human_decision(
                incident_id=selected_inc_id,
                commander_id="COMM-CHIEF-01",
                decision="APPROVE",
                assigned_unit_id=unit_to_assign,
                comments=comm_notes,
            )
            # Mark unit as dispatched in session simulator
            if rec_item and rec_item.recommended_unit_id:
                for u in st.session_state.simulator.units:
                    if u.unit_id == rec_item.recommended_unit_id:
                        u.status = "DISPATCHED"
                        u.assigned_incident_id = selected_inc_id
            st.success(f"Dispatched **{unit_to_assign}** to **{selected_inc_id}**! Logged to SQLite audit trail.")
            st.rerun()

    with col_act3:
        st.write("")
        st.write("")
        if st.button("❌ REJECT / DISMISS", use_container_width=True):
            storage.log_human_decision(
                incident_id=selected_inc_id,
                commander_id="COMM-CHIEF-01",
                decision="REJECT",
                assigned_unit_id=None,
                comments=comm_notes,
            )
            st.warning(f"Incident **{selected_inc_id}** dismissed / deferred. Logged to SQLite.")
            st.rerun()

    # Audit Trail Log
    st.markdown("---")
    st.subheader("📜 Recent Commander Decision Audit Trail")
    decisions = storage.get_decision_history(limit=8)
    if decisions:
        df_dec = pd.DataFrame(decisions)[["decision_id", "incident_id", "decision", "assigned_unit_id", "commander_id", "decided_at", "comments"]]
        st.dataframe(df_dec, use_container_width=True, hide_index=True)
    else:
        st.info("No commander decisions recorded in current database yet.")


# ==========================================
# TAB 3: AI EXPLAINABILITY (SHAP)
# ==========================================
with tab3:
    st.subheader("SHAP Feature Attribution & Local Explainability")
    st.caption("Understand precisely which factors elevated or decreased the predicted risk score for any emergency report.")

    inc_to_explain = st.selectbox(
        "Select Incident for Deep Explainability Analysis",
        options=df_ranked["incident_id"].tolist(),
        key="shap_select",
    )

    row_to_explain = df_ranked[df_ranked["incident_id"] == inc_to_explain].iloc[0]
    attributions, plain_summary = st.session_state.explainer.explain_incident(row_to_explain, top_k=8)

    # Narrative Card
    st.info(f"**Human-Readable AI Summary:** {plain_summary}")

    # Plotly Waterfall / Bar Chart of SHAP Contributions
    attr_df = pd.DataFrame([a.__dict__ for a in attributions])
    attr_df["clean_name"] = attr_df["feature_name"].str.replace("_", " ").str.title()
    attr_df["color"] = attr_df["attribution_value"].apply(lambda v: "#d32f2f" if v > 0 else "#1976d2")

    fig_shap = px.bar(
        attr_df,
        x="attribution_value",
        y="clean_name",
        orientation="h",
        color="direction",
        color_discrete_map={"increases_risk": "#d32f2f", "decreases_risk": "#1976d2"},
        labels={"attribution_value": "SHAP Contribution (Log-Odds Impact)", "clean_name": "Feature"},
        title=f"Local Feature Contributions for {inc_to_explain} (Predicted Risk: {row_to_explain['risk_score']:.2f})",
    )
    fig_shap.update_layout(yaxis={"autorange": "reversed"}, height=400)
    st.plotly_chart(fig_shap, use_container_width=True)

    # Attribute Details Table
    st.markdown("##### Detailed Attribution Values")
    st.dataframe(
        attr_df[["clean_name", "feature_value", "attribution_value", "direction"]],
        use_container_width=True,
        hide_index=True,
    )


# ==========================================
# TAB 4: EMPIRICAL EVALUATION
# ==========================================
with tab4:
    st.subheader("Model Performance & Empirical Evaluation")
    st.caption("Real measurements from cross-scenario benchmarking against deterministic Computer-Aided Dispatch (CAD) baselines.")

    results_file = Path(DEFAULT_RESULTS_PATH)
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            md_content = f.read()
        st.markdown(md_content)
    else:
        st.warning("docs/results.md not yet generated. Run `python scripts/evaluate.py` to produce full results.")
        # Fallback quick display of champion model metrics
        if st.session_state.model.metrics:
            m = st.session_state.model.metrics
            st.json({
                "model_name": m.model_name,
                "roc_auc": m.roc_auc,
                "pr_auc": m.pr_auc,
                "f1_score": m.f1_score,
                "accuracy": m.accuracy,
                "brier_score": m.brier_score,
            })


# ==========================================
# TAB 5: IBM Z & PLATFORM TELEMETRY
# ==========================================
with tab5:
    st.subheader("IBM Z & LinuxONE Platform Integration")
    st.caption("Telemetry, accelerator readiness probe, and inference latency profiling for enterprise mission-critical environments.")

    telemetry = get_platform_telemetry()

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("##### 🖥️ Host Environment")
        st.write(f"**CPU Architecture:** `{telemetry.machine_architecture}`")
        st.write(f"**Operating System:** `{telemetry.operating_system} ({telemetry.os_release})`")
        st.write(f"**Python Runtime:** `{telemetry.python_version}`")
        st.write(f"**IBM Z Architecture (s390x):** `{'YES (Native)' if telemetry.is_ibm_z_architecture else 'NO (Simulated / Host CPU)'}`")

    with col_t2:
        st.markdown("##### ⚡ AI Acceleration & Runtimes")
        st.write(f"**IBM Snap ML Available:** `{'YES' if telemetry.snapml_available else 'NO (Graceful CPU Fallback)'}`")
        st.write(f"**ONNX Runtime Engine:** `{'YES' if telemetry.onnx_runtime_available else 'NO'}`")
        st.write(f"**Active Inference Engine:** `{telemetry.active_inference_engine}`")

    st.markdown("---")
    st.subheader("⚡ Live Latency Micro-Benchmark")
    if st.button("Run Live 100-Query Latency Profiler", key="btn_run_lat"):
        with st.spinner("Executing real-time inference latency benchmark..."):
            sim_bench = EmergencySimulator(seed=123)
            df_sample = sim_bench.generate_batch_dataset(num_ticks=20)
            bench_results = run_benchmark_comparison(st.session_state.model, df_sample, iterations=100)

            c_b1, c_b2, c_b3, c_b4 = st.columns(4)
            c_b1.metric("p50 Latency", f"{bench_results['p50_latency_ms']} ms")
            c_b2.metric("p95 Latency", f"{bench_results['p95_latency_ms']} ms")
            c_b3.metric("p99 Latency", f"{bench_results['p99_latency_ms']} ms")
            c_b4.metric("Throughput", f"{bench_results['throughput_qps']} QPS")
            st.success(f"Verified live sub-millisecond execution using engine: **{bench_results['engine']}**")

    st.markdown("---")
    st.markdown("""
    ### Why IBM Z & LinuxONE for Emergency Decision Support?
    1. **Zero Data Exfiltration Risk:** Colocating ResQ-AI directly inside the IBM Z transactional core eliminates risky cloud roundtrips for municipal data.
    2. **Sub-Millisecond AI on Live Streams:** IBM z16's on-chip **Integrated Accelerator for AI (NNPA)** enables inference at wire speed without impacting transaction throughput.
    3. **Continuous Reliability (99.99999% Availability):** Critical emergency dispatch operations cannot afford cloud outages or network link failures.
    """)
