"""Unit tests for ML model training, evaluation, champion selection, and prediction."""

import numpy as np
import pytest
from resq.models import ResQRiskModel, train_and_compare_models
from resq.simulator import EmergencySimulator


def test_train_and_compare_models(tmp_path):
    """Verify multi-model comparison selects a valid champion and serializes properly."""
    sim = EmergencySimulator(seed=42)
    df = sim.generate_batch_dataset(num_ticks=25)

    champion, metrics_list = train_and_compare_models(df, test_ratio=0.30, seed=42)
    assert champion is not None
    assert len(metrics_list) >= 2  # At least LogReg and RandomForest

    for m in metrics_list:
        assert 0.0 <= m.roc_auc <= 1.0
        assert 0.0 <= m.pr_auc <= 1.0
        assert 0.0 <= m.accuracy <= 1.0

    # Test prediction
    sample_df = df.iloc[:3]
    probs = champion.predict_risk_proba(sample_df)
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs)

    # Test save and reload
    save_path = champion.save(str(tmp_path))
    reloaded = ResQRiskModel.load(save_path)
    reloaded_probs = reloaded.predict_risk_proba(sample_df)
    np.testing.assert_allclose(probs, reloaded_probs, rtol=1e-5)
