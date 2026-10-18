"""Training script: simulates training data, compares 4 ML models, and persists champion artifact."""

from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from resq.config import DEFAULT_MODELS_DIR, DEFAULT_SEED_DATA_PATH, RANDOM_SEED
from resq.models import train_and_compare_models
from resq.simulator import EmergencySimulator


def main():
    print("=" * 70)
    print("ResQ-AI: ML Model Training & Selection Pipeline")
    print("IBM Z Datathon 2026 - Real-Time AI for Critical Decisions")
    print("=" * 70)

    # 1. Simulate training data across diverse scenarios
    print("\n[Step 1/4] Generating realistic synthetic incident stream...")
    sim = EmergencySimulator(seed=RANDOM_SEED)
    df_raw = sim.generate_batch_dataset(num_ticks=150)
    print(f"Generated {len(df_raw)} incidents across 150 simulation ticks.")
    print(f"Class distribution: {df_raw['is_high_risk'].value_counts().to_dict()}")

    # 2. Persist seed sample dataset
    data_path = Path(DEFAULT_SEED_DATA_PATH)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(data_path, index=False)
    print(f"[Step 2/4] Saved seed dataset ({len(df_raw)} rows) to: {data_path}")

    # 3. Train and compare 4 models (LogReg, Random Forest, XGBoost, LightGBM)
    print("\n[Step 3/4] Training candidate models with temporal train/test split (no leakage)...")
    champion, metrics_list = train_and_compare_models(df_raw, test_ratio=0.25, seed=RANDOM_SEED)

    print("\n" + "-" * 75)
    print(f"{'Model':<20} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'F1-Score':<9} | {'Accuracy':<9} | {'Brier':<9}")
    print("-" * 75)
    for m in metrics_list:
        print(f"{m.model_name:<20} | {m.roc_auc:<9.4f} | {m.pr_auc:<9.4f} | {m.f1_score:<9.4f} | {m.accuracy:<9.4f} | {m.brier_score:<9.4f}")
    print("-" * 75)

    # 4. Save champion model
    print("\n[Step 4/4] Persisting Champion Model...")
    saved_path = champion.save(DEFAULT_MODELS_DIR)
    print(f"Successfully serialized Champion Model ({champion.model_name}) to: {saved_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
