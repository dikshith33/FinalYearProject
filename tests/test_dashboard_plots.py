"""Unit tests for the Dark Dashboard Plot Generator."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from satsim.visualization.dashboard_plots import (
    plot_base_paper_vs_our_gat_radar,
    plot_architecture_feature_scope_bar,
    plot_lstm_actual_vs_predicted,
)


class TestDashboardPlots(unittest.TestCase):
    """Test suite for validating generated dashboard visualizations."""

    def test_plot_base_paper_vs_our_gat_radar(self) -> None:
        """Verify radar chart generation and file creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "radar_test.png"
            plot_base_paper_vs_our_gat_radar(out_file)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 1000)

    def test_plot_architecture_feature_scope_bar(self) -> None:
        """Verify architecture bar chart generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "bar_test.png"
            plot_architecture_feature_scope_bar(out_file)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 1000)

    def test_plot_lstm_actual_vs_predicted(self) -> None:
        """Verify LSTM actual vs predicted tracking plot generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "tracking_test.png"
            plot_lstm_actual_vs_predicted(out_file)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
