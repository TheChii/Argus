"""Tests for Macro pipeline and feature calculation."""
import tempfile
from pathlib import Path

import pandas as pd

from src.data.macro import MacroPipeline


def test_macro_features_calculation():
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MacroPipeline(macro_dir=Path(tmpdir))

        # Create mock synthetic macro series dataframe
        dates = pd.date_range("2026-09-01", periods=10, freq="B").strftime("%Y-%m-%d")
        mock_data = pd.DataFrame({
            "date": dates,
            "T10Y2Y": [0.20 + i * 0.02 for i in range(10)],
            "DFF": [5.25 + i * 0.01 for i in range(10)],
            "VIXCLS": [14.0 + i * 0.5 for i in range(10)],
            "BAMLH0A0HYM2": [3.20 + i * 0.05 for i in range(10)],
        })

        features = pipeline.get_macro_features(mock_data)

        assert "yield_spread" in features.columns
        assert "fed_funds" in features.columns
        assert "fed_funds_change_5d" in features.columns
        assert "vix" in features.columns
        assert "vix_change_5d" in features.columns
        assert "hy_spread" in features.columns

        # Row 5 (index 5) should have 5-day difference
        expected_vix_diff = mock_data["VIXCLS"].iloc[5] - mock_data["VIXCLS"].iloc[0]
        actual_vix_diff = features["vix_change_5d"].iloc[5]
        assert abs(expected_vix_diff - actual_vix_diff) < 1e-6
