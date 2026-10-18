# IBM Z and LinuxONE Enterprise Deployment Plan

**Challenge:** Real-Time AI for Critical Decisions  
**Target Platform:** IBM z16 / LinuxONE (s390x architecture)  
**Status Disclosure:** Verification on native s390x hardware is **PLANNED**. Current testing and benchmarks were executed on host multi-core CPU architecture with software emulation and fallback probes.

---

## 1. Why IBM Z / LinuxONE for Emergency Decision Support?

Emergency dispatch systems are quintessential mission-critical workloads. In large urban municipalities, emergency response infrastructure must operate without interruption under catastrophic regional grid failures.

| Enterprise Driver | The Challenge | The IBM Z / LinuxONE Advantage |
| :--- | :--- | :--- |
| **Colocated Low Latency** | Network latency to third-party public clouds can take 150–350 ms, delaying triage. | Models run in the **same memory bus and LPAR** as the core transactional Computer-Aided Dispatch (CAD) database. Inference takes < 1 ms. |
| **Integrated AI Acceleration** | Heavy deep learning or ensemble scoring consumes general CPU cycles needed for dispatch. | The **IBM Telum processor** includes an on-chip **Integrated Accelerator for AI (NNPA)** delivering up to 6 TFLOPS per chip for wire-speed scoring. |
| **High Availability** | Cloud region outages paralyze municipal emergency response centers. | IBM Z provides **99.99999% availability** (less than 3.15 seconds of unplanned downtime per year) with hot-swappable redundant hardware. |
| **Pervasive Security & Privacy** | Emergency caller medical details and 911 audio are prime targets for cyberattacks. | **Pervasive Encryption** encrypts data at rest, in flight, and in use (Confidential Computing) with dedicated cryptographic coprocessors (CPACF). |

---

## 2. IBM Snap ML Acceleration Integration

ResQ-AI incorporates direct compatibility with **IBM Snap ML** (`snapml`), an enterprise machine learning library optimized for IBM hardware:

1. **Tree Ensemble Acceleration:** Snap ML optimizes gradient boosted decision trees and random forests, achieving up to a **4x–10x latency reduction** compared to standard CPU runtimes.
2. **Telum NNPA Exploitation:** On supported IBM z16 and LinuxONE III/IV environments running s390x Linux, Snap ML automatically offloads matrix and tree operations to the on-chip AI accelerator.
3. **Graceful Fallback Pipeline:** As implemented in [`src/resq/ibmz.py`](../src/resq/ibmz.py), ResQ-AI probes for `snapml` at runtime. If present, it engages the accelerated engine; otherwise, it seamlessly falls back to scikit-learn / XGBoost native runtimes with zero user intervention.

---

## 3. Step-by-Step Deployment Guide for LinuxONE (s390x)

The following reproducible steps configure ResQ-AI on a Red Hat Enterprise Linux (RHEL 9) or Ubuntu 22.04 LTS instance running on an **s390x** LPAR:

### Step 1: Provision an s390x Instance
Access an IBM LinuxONE Community Cloud instance or enterprise z/VM guest:
```bash
# Verify s390x hardware architecture
uname -m
# Expected output: s390x
```

### Step 2: System Prerequisites
```bash
# On RHEL 9 (s390x):
sudo dnf update -y
sudo dnf install -y python3.11 python3.11-devel python3.11-pip git gcc gcc-c++ make

# On Ubuntu 22.04 (s390x):
sudo apt-get update -y
sudo apt-get install -y python3-dev python3-pip git build-essential
```

### Step 3: Clone ResQ-AI and Set Up Virtual Environment
```bash
git clone https://github.com/resq-ai/resq-ai.git
cd resq-ai

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Step 4: Install Dependencies & IBM Snap ML (s390x)
```bash
# Install core requirements
pip install -r requirements.txt

# Install IBM Snap ML on s390x (pre-compiled wheel for Linux on Z)
pip install snapml
```

### Step 5: Verify Hardware Telemetry & Run Benchmark
```bash
# Run latency profiler to confirm Snap ML activation on s390x
python scripts/benchmark_latency.py
```
Expected output on LinuxONE:
```text
[Platform Information]
  Architecture:     s390x
  IBM Z Detected:   True
  Snap ML Active:   True
  Active Engine:    IBM Snap ML on IBM Z Integrated Accelerator for AI (NNPA)
```

### Step 6: Launch Full Stack
```bash
python run_demo.py
```

---

## 4. Planned Hardware Validation Roadmap

| Phase | Milestone | Target Infrastructure | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Local prototype development and simulation | x86_64 / Apple Silicon | **Completed & Verified** |
| **Phase 2** | Automated ONNX and Snap ML fallback probe | Multi-core CPU | **Completed & Verified** |
| **Phase 3** | LinuxONE Community Cloud environment deployment | s390x (Ubuntu 22.04) | **Planned** |
| **Phase 4** | Native IBM z16 Telum NNPA micro-benchmark | z16 LPAR (RHEL 9) | **Planned** |
