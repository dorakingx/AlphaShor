"""Sanity tests for textbook QPE path (mock phase oracle)."""

import unittest
import numpy as np

from main import MockPhaseOracle, QSPPhaseEstimator


class TestStandardQPE(unittest.TestCase):
    def test_mock_phase_gate_2pi_third(self):
        theta = (2.0 / 3.0) * np.pi
        oracle = MockPhaseOracle(phase=theta)
        est = QSPPhaseEstimator(oracle, degree=5, shots=8000, error_rate=0.0)
        ph = est.estimate_phase_standard_qpe(precision_bits=8)
        err = abs(ph - theta)
        err = min(err, 2 * np.pi - err)
        self.assertLess(
            err,
            0.25,
            msg=f"QPE phase error too large: est={ph}, expected={theta}, err={err}",
        )


if __name__ == "__main__":
    unittest.main()
