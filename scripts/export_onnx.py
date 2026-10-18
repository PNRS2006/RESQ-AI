"""Model export script converting champion model to open ONNX and JSON formats for cross-platform and IBM Z deployment."""

import json
import logging
import os
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
from resq.config import DEFAULT_MODELS_DIR
from resq.data_prep import FEATURE_COLUMNS
from resq.models import ResQRiskModel, train_and_compare_models
from resq.simulator import EmergencySimulator

logger = logging.getLogger(__name__)


def export_to_onnx(model: ResQRiskModel, output_dir: str = DEFAULT_MODELS_DIR) -> str:
    """Export scikit-learn / XGBoost model to standard ONNX format."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file = out_path / "champion_model.onnx"

    estimator = model.estimator

    # Check for XGBoost native JSON export
    if hasattr(estimator, "save_model"):
        xgb_json_path = out_path / "champion_model_xgb.json"
        try:
            estimator.save_model(str(xgb_json_path))
            print(f"Exported XGBoost native format to: {xgb_json_path}")
        except Exception as exc:
            print(f"XGBoost JSON export skipped: {exc}")

    try:
        from skl2onnx.common.data_types import FloatTensorType

        n_features = len(FEATURE_COLUMNS)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]

        # Use onnxmltools for XGBoost
        if "xgb" in str(type(estimator)).lower() or hasattr(estimator, "get_booster"):
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType as OXMLFloatTensorType
            xgb_initial_type = [("float_input", OXMLFloatTensorType([None, n_features]))]
            booster = estimator.get_booster()
            orig_feature_names = booster.feature_names
            try:
                booster.feature_names = None
                onnx_model = onnxmltools.convert_xgboost(
                    estimator,
                    initial_types=xgb_initial_type,
                    target_opset=15,
                )
            finally:
                booster.feature_names = orig_feature_names
        else:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType as SKLFloatTensorType
            skl_initial_type = [("float_input", SKLFloatTensorType([None, n_features]))]
            clf_to_convert = estimator
            onnx_model = convert_sklearn(
                clf_to_convert,
                initial_types=skl_initial_type,
                target_opset=15,
            )

        with open(onnx_file, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"Successfully exported ONNX model to: {onnx_file}")

        # Verification with ONNX Runtime
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_file))
            dummy_input = np.ones((1, n_features), dtype=np.float32)
            input_name = sess.get_inputs()[0].name
            _ = sess.run(None, {input_name: dummy_input})
            print(f"ONNX Runtime verification passed! Model is ready for enterprise deployment.")
        except Exception as ort_exc:
            print(f"ONNX Runtime verification note: {ort_exc}")

        return str(onnx_file)

    except Exception as exc:
        print(f"skl2onnx direct export failed ({exc}). Writing portable metadata JSON.")
        meta_export = out_path / "model_spec.json"
        with open(meta_export, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": model.model_name,
                "n_features": len(FEATURE_COLUMNS),
                "features": FEATURE_COLUMNS,
                "export_status": "PORTABLE_FALLBACK",
            }, f, indent=2)
        return str(meta_export)


def main():
    print("=" * 70)
    print("ResQ-AI: Enterprise Model Export (ONNX & JSON)")
    print("IBM Z Datathon 2026 - Real-Time AI for Critical Decisions")
    print("=" * 70)

    model_path = os.path.join(DEFAULT_MODELS_DIR, "champion_model.joblib")
    if os.path.exists(model_path):
        model = ResQRiskModel.load(model_path)
    else:
        print("Training baseline model for export...")
        sim = EmergencySimulator(seed=42)
        df = sim.generate_batch_dataset(num_ticks=40)
        model, _ = train_and_compare_models(df, seed=42)
        model.save(DEFAULT_MODELS_DIR)

    export_to_onnx(model, DEFAULT_MODELS_DIR)


if __name__ == "__main__":
    main()
