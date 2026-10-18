"""IBM Z and LinuxONE platform telemetry, IBM Snap ML accelerator probe, and inference interface."""

from dataclasses import asdict, dataclass
import logging
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check for IBM Snap ML library (accelerated inference on IBM z16 / LinuxONE)
try:
    import snapml
    HAS_SNAPML = True
    SNAPML_VERSION = getattr(snapml, "__version__", "unknown")
except ImportError:
    HAS_SNAPML = False
    SNAPML_VERSION = None

# Check for ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
    ONNX_RUNTIME_VERSION = getattr(ort, "__version__", "unknown")
except ImportError:
    HAS_ONNX_RUNTIME = False
    ONNX_RUNTIME_VERSION = None


@dataclass
class PlatformTelemetry:
    """Hardware and runtime telemetry for enterprise deployment monitoring."""
    machine_architecture: str
    operating_system: str
    os_release: str
    python_version: str
    is_ibm_z_architecture: bool
    snapml_available: bool
    snapml_version: Optional[str]
    onnx_runtime_available: bool
    onnx_runtime_version: Optional[str]
    active_inference_engine: str
    hardware_notes: str


def get_platform_telemetry() -> PlatformTelemetry:
    """Probe hardware platform, CPU architecture, and AI acceleration libraries."""
    mach = platform.machine().lower()
    is_z = mach in ("s390x", "s390")

    if is_z and HAS_SNAPML:
        active_engine = "IBM Snap ML on IBM Z Integrated Accelerator for AI (NNPA)"
        notes = "Running natively on IBM Z / LinuxONE enterprise hardware with on-chip AI acceleration."
    elif HAS_SNAPML:
        active_engine = f"IBM Snap ML v{SNAPML_VERSION} (CPU Emulation / Acceleration)"
        notes = "Snap ML library detected and active on host CPU."
    elif HAS_ONNX_RUNTIME:
        active_engine = f"ONNX Runtime v{ONNX_RUNTIME_VERSION} (Standard CPU Execution Provider)"
        notes = "ONNX Runtime available for cross-platform model execution."
    else:
        active_engine = "Standard scikit-learn / XGBoost Native CPU Runtime"
        notes = "Host CPU execution without proprietary accelerators."

    return PlatformTelemetry(
        machine_architecture=platform.machine(),
        operating_system=platform.system(),
        os_release=platform.release(),
        python_version=platform.python_version(),
        is_ibm_z_architecture=is_z,
        snapml_available=HAS_SNAPML,
        snapml_version=SNAPML_VERSION,
        onnx_runtime_available=HAS_ONNX_RUNTIME,
        onnx_runtime_version=ONNX_RUNTIME_VERSION,
        active_inference_engine=active_engine,
        hardware_notes=notes,
    )


def run_benchmark_comparison(
    model: Any,
    X_sample: pd.DataFrame,
    iterations: int = 100,
) -> Dict[str, Any]:
    """Measure real empirical inference latency (p50, p90, p95, p99) and QPS throughput."""
    if len(X_sample) == 0:
        return {"error": "Empty sample"}

    latencies_ms: List[float] = []

    # Warmup
    for _ in range(5):
        _ = model.predict_proba(X_sample.iloc[:1])

    # Benchmark individual single-incident scoring (simulating real-time incoming CAD alerts)
    single_record = X_sample.iloc[:1]
    t_start = time.perf_counter()

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict_proba(single_record)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    total_time_sec = time.perf_counter() - t_start
    qps = round(iterations / max(1e-5, total_time_sec), 1)

    lat_arr = np.array(latencies_ms)
    return {
        "iterations": iterations,
        "p50_latency_ms": round(float(np.percentile(lat_arr, 50)), 3),
        "p90_latency_ms": round(float(np.percentile(lat_arr, 90)), 3),
        "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 3),
        "mean_latency_ms": round(float(np.mean(lat_arr)), 3),
        "min_latency_ms": round(float(np.min(lat_arr)), 3),
        "throughput_qps": qps,
        "engine": get_platform_telemetry().active_inference_engine,
    }
