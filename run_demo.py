"""One-command full-lifecycle launcher for ResQ-AI.

Generates training data, trains and compares 4 ML models, runs cross-scenario
evaluations, benchmarks latency, exports to ONNX, starts the FastAPI service
in the background, and launches the interactive Streamlit command center.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Ensure resq package in python path
repo_root = Path(__file__).resolve().parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def print_banner():
    banner = """
================================================================================
  ____                 ___             _    ___ 
 |  _ \  ___  ___  __ / _ \           / \  |_ _|
 | |_) |/ _ \/ __|/ _ \ (_) | _____  / _ \  | | 
 |  _ <|  __/\__ \ (_) \__, |_____|/ ___ \ | | 
 |_| \_\\___||___/\__\_\ /_/      /_/   \_\___|

 Real-Time AI Emergency Decision-Support System
 Challenge: "Real-Time AI for Critical Decisions" (IBM Z Datathon 2026)
================================================================================
 [!] Decision support only. A human approves every dispatch. Data is simulated.
================================================================================
"""
    print(banner)


def run_command_with_status(description: str, cmd_args: list):
    """Run a Python command with clear progress logging."""
    print(f"\n[+] {description}...")
    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable] + cmd_args,
        cwd=str(repo_root),
        env=env,
        capture_output=False,
    )
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"[-] Error executing {description} (exit code: {result.returncode})")
        sys.exit(result.returncode)
    print(f"[✓] {description} completed in {elapsed:.2f} seconds.")


def start_fastapi_server():
    """Run FastAPI server in a background thread."""
    import uvicorn
    from resq.api import app

    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server.run()


def main():
    print_banner()

    # Step 1: Train & select champion model
    run_command_with_status(
        "Step 1/5: Training ML Models & Champion Selection",
        ["scripts/train.py"],
    )

    # Step 2: Cross-scenario evaluation against CAD baseline
    run_command_with_status(
        "Step 2/5: Cross-Scenario Rigorous Evaluation (Writing docs/results.md)",
        ["scripts/evaluate.py"],
    )

    # Step 3: Latency micro-benchmarking
    run_command_with_status(
        "Step 3/5: Real-Time Inference Latency Benchmarking",
        ["scripts/benchmark_latency.py"],
    )

    # Step 4: ONNX & Enterprise Model Export
    run_command_with_status(
        "Step 4/5: Exporting Champion Model to ONNX & JSON",
        ["scripts/export_onnx.py"],
    )

    # Step 5: Start FastAPI Background Service
    print("\n[+] Step 5/5: Starting Services...")
    print("    Starting FastAPI REST Service on http://127.0.0.1:8000 ...")
    api_thread = threading.Thread(target=start_fastapi_server, daemon=True)
    api_thread.start()
    time.sleep(2)  # Give API a moment to bind

    print("\n" + "=" * 70)
    print("  🚀 ALL SERVICES READY & VERIFIED!")
    print("  FastAPI Documentation:  http://127.0.0.1:8000/docs")
    print("  FastAPI Health Probe:   http://127.0.0.1:8000/health")
    print("  FastAPI Platform Info:  http://127.0.0.1:8000/platform")
    print("  Streamlit Dashboard:    http://localhost:8501")
    print("=" * 70 + "\n")
    print("[+] Launching Streamlit Command Center Dashboard...")

    # Launch Streamlit in foreground
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/dashboard.py",
            "--server.port=8501",
            "--server.headless=true",
        ],
        cwd=str(repo_root),
        env=env,
    )


if __name__ == "__main__":
    main()
