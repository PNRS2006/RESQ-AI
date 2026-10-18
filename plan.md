# ResQ-AI Implementation Plan

**Project:** ResQ-AI — AI-Assisted Emergency Decision-Support System  
**Hackathon:** IBM Z Datathon 2026 (Challenge: *"Real-Time AI for Critical Decisions"*)  
**Core Mission:** Rapid risk scoring, prioritized incident ranking, and congestion-aware resource dispatch recommendations with strict Human-in-the-Loop approval.

---

## 1. Objectives & Architectural Principles

- **Human-in-the-Loop Guarantee:** The AI acts strictly as an advisory decision-support system. It never dispatches units autonomously; every action requires explicit human incident commander sign-off.
- **Reproducibility & Zero API Key Friction:** Operates completely out-of-the-box on a laptop with standard Python 3.10+ and a single entrypoint command (`python run_demo.py`).
- **Complete Transparency & Honest Metrics:** Simulated datasets and synthetic risk formulation are clearly declared in documentation. Zero hallucinated or hardcoded metrics; every single number in reports, README, and dashboard is generated dynamically by real code executions.
- **Resilient Fallbacks:** Redis Streams falls back gracefully to in-memory FIFO queues, Open-Meteo live weather falls back to dynamic meteorological simulation, and IBM Snap ML falls back cleanly to scikit-learn/XGBoost/LightGBM CPU runtimes.

---

## 2. Directory Layout

```text
resq-ai/
├── README.md                          # Comprehensive project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Pinned core dependencies
├── .gitignore                         # Python, venv, caches, logs
├── run_demo.py                        # One-command full-lifecycle demo launcher
├── plan.md                            # Detailed architectural roadmap
├── src/
│   └── resq/
│       ├── __init__.py
│       ├── config.py                  # City bounding boxes (Delhi default), weights, thresholds
│       ├── simulator.py               # Time-tick incident, weather, traffic, & unit generator
│       ├── data_prep.py               # Cleaning, range validation, temporal splitting, feature engineering
│       ├── weather.py                 # Open-Meteo live API integration + realistic simulation fallback
│       ├── models.py                  # Multi-model training (LogReg, RF, XGB, LGBM), selection & inference
│       ├── ranking.py                 # Dynamic urgency ranking & rule-based baseline comparator
│       ├── allocation.py              # Nearest suitable unit recommendation, Haversine + traffic ETA
│       ├── explain.py                 # Local SHAP feature attributions & plain-English explanations
│       ├── stream.py                  # Redis Streams producer/consumer + robust in-memory fallback
│       ├── storage.py                 # SQLite engine (incidents, recommendations, logged decisions)
│       ├── api.py                     # High-performance FastAPI REST service with Pydantic schemas
│       └── ibmz.py                    # ONNX / Snap ML runtime probe, platform telemetry & hardware info
├── app/
│   └── dashboard.py                   # 5-tab Streamlit Command Center (Folium, Plotly, Decision audit)
├── scripts/
│   ├── train.py                       # Trains, evaluates, and persists the champion model
│   ├── evaluate.py                    # Evaluates models across scenarios, writes docs/results.md
│   ├── benchmark_latency.py           # Measures real inference latency (p50/p95/throughput)
│   └── export_onnx.py                 # Exports trained model to ONNX & tree JSON formats
├── tests/
│   ├── test_simulator.py
│   ├── test_data_prep.py
│   ├── test_models.py
│   ├── test_ranking.py
│   ├── test_allocation.py
│   ├── test_storage.py
│   ├── test_stream.py
│   └── test_api.py
├── docs/
│   ├── ARCHITECTURE.md                # System topology, dataflow, and Mermaid diagrams
│   ├── DATA.md                        # Synthetic dataset specification, formulas, & real data mapping
│   ├── IBM_Z_PLAN.md                  # LinuxONE (s390x) enterprise integration, Telum AI accelerator plan
│   ├── results.md                     # Auto-generated benchmark metrics & scenario comparisons
│   └── screenshots/                   # Captured screenshots of the live dashboard
└── data/
    └── sample_incidents.csv           # Small generated seed dataset (<20MB repo limit)
```

---

## 3. Step-by-Step Implementation Roadmap

### Phase 1: Environment & Project Scaffolding
- [x] Create project structure in `resq-ai/`.
- [ ] Create `requirements.txt` containing compatible dependencies (pandas, numpy, scikit-learn, xgboost, lightgbm, shap, fastapi, uvicorn, streamlit, folium, streamlit-folium, plotly, requests, onnx, skl2onnx, pytest).
- [ ] Create `.gitignore` and `LICENSE` (MIT).
- [ ] Set up virtual environment and verify Python package installations.

### Phase 2: Configuration, Simulation & Data Preparation
- [ ] **`src/resq/config.py`**: Bounding boxes for Delhi (28.40-28.88 N, 76.84-77.35 E), incident types (road accident, fire, flood, medical emergency, building collapse), resource unit types (ambulance, fire engine, police cruiser), default weights for urgency ranking.
- [ ] **`src/resq/weather.py`**: Query Open-Meteo REST endpoint (lat/lon, temp, rain, wind, visibility); fallback seamlessly to dynamic sinusoidal/stochastic weather simulator on network disconnect or rate limit.
- [ ] **`src/resq/simulator.py`**: Tick-based simulation generating incidents, traffic congestion index (0-1), resource unit locations and statuses. Pre-configured scenarios:
  1. *Normal day*
  2. *Heavy rain + floods*
  3. *Multi-vehicle highway accident*
  4. *City-wide surge (mass casualty incident)*
- [ ] **`src/resq/data_prep.py`**: Cleans raw feeds, validates bounding boxes, engineers temporal and environmental features (e.g. rain_x_congestion, casualty_severity_ratio), and performs time-series split (no forward-looking data leakage).

### Phase 3: Modeling, Ranking, Allocation & Explainability
- [ ] **`src/resq/models.py`**: Model training pipeline evaluating Logistic Regression, Random Forest, XGBoost, and LightGBM. Automatically selects best model based on real validation ROC-AUC / PR-AUC and serializes model artifacts.
- [ ] **`src/resq/ranking.py`**: Computes Urgency Score combining ML Risk Score, casualty count, and elapsed time since incident report. Implements fixed-rule baseline (severity-first heuristic) for fair side-by-side benchmarking.
- [ ] **`src/resq/allocation.py`**: Recommends nearest available suitable unit matching incident type using Haversine distance and congestion-adjusted ETA ($ETA = \frac{distance}{speed \times (1 - 0.7 \times congestion)}$). Explicitly tracks unmet demand when units are exhausted.
- [ ] **`src/resq/explain.py`**: TreeSHAP / KernelSHAP integration extracting top feature attributions per incident, translating them into human-readable dispatch explanations.

### Phase 4: Streaming & Persistence
- [ ] **`src/resq/stream.py`**: Event-driven streaming abstraction supporting Redis Streams when available, seamlessly falling back to a thread-safe in-memory queue.
- [ ] **`src/resq/storage.py`**: SQLite database managing `incidents`, `recommendations`, and `decisions` (audit trail with timestamps, commander IDs, approved/rejected actions, and feedback notes).

### Phase 5: REST API & IBM Z / LinuxONE Acceleration
- [ ] **`src/resq/api.py`**: FastAPI service providing:
  - `POST /predict`: Real-time risk scoring and explanation generation
  - `GET /incidents`: Current ranked incident queue
  - `GET /recommendations`: Resource dispatch proposals
  - `POST /decisions`: Human commander approval/rejection logging
  - `GET /platform`: Machine architecture (`platform.machine()`), Python runtime, Snap ML / ONNX status
  - `GET /health`: System diagnostics
- [ ] **`src/resq/ibmz.py`**: Telemetry and inference engine detecting IBM z16/LinuxONE s390x environment, testing for `snapml`, or running ONNX / CPU fallbacks.
- [ ] **`scripts/export_onnx.py`**: Exports trained ensemble to ONNX format.
- [ ] **`scripts/benchmark_latency.py`**: Runs real micro-benchmarks measuring p50, p90, p95, p99 latency and QPS throughput.

### Phase 6: Scripts & Full Evaluation
- [ ] **`scripts/train.py`**: End-to-end dataset generation, multi-model training, model selection, and artifact persistence.
- [ ] **`scripts/evaluate.py`**: Evaluates champion model and rule-based baseline across all 4 scenarios with multiple random seeds. Outputs metrics (ROC-AUC, PR-AUC, NDCG@k, Top-k Recall, latency) and writes `docs/results.md`.

### Phase 7: Interactive Streamlit Command Center
- [ ] **`app/dashboard.py`**:
  - Banner: *"Decision support only. A human approves every action. Data is simulated."*
  - **Tab 1: Live Command View**: Interactive Folium map with color-coded risk markers, emergency response units, scenario selector, next-tick trigger, and live alert feed.
  - **Tab 2: Priority Queue & Dispatch**: Ranked incident table with ML urgency scores, SHAP summary reasons, recommended unit, ETA, and interactive **Approve / Reject** buttons storing decisions into SQLite.
  - **Tab 3: AI Explainability (SHAP)**: Waterfall and feature importance charts explaining individual incident scores.
  - **Tab 4: Performance & Benchmarking**: Real scenario metrics, AI vs baseline comparison charts, ROC/PR curves.
  - **Tab 5: IBM Z & Platform Telemetry**: Architecture specifications, Snap ML status, and real benchmark latency figures.

### Phase 8: Comprehensive Testing & Verification
- [ ] Unit tests for all modules in `tests/` covering simulator, data prep, models, ranking, allocation, storage, stream, and FastAPI endpoints.
- [ ] Execute `pytest -v` and ensure 100% test pass rate.
- [ ] Execute `scripts/train.py`, `scripts/evaluate.py`, `scripts/export_onnx.py`, and `scripts/benchmark_latency.py`.
- [ ] Execute `run_demo.py` to launch API backend and Streamlit dashboard.
- [ ] Verify each tab in the browser using browser tools, capture screenshots into `docs/screenshots/`.
- [ ] Initialize Git repository, configure `.gitignore`, and record structured commits.
