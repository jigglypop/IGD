import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.experiment import run_experiment_a, run_experiment_b, sanity_energy


class TestPivot(unittest.TestCase):
    def test_dirichlet_energy_nonnegative(self):
        e = sanity_energy(n=64)
        self.assertGreaterEqual(e, -1e-6)

    def test_experiment_a_improves_with_lbo(self):
        r = run_experiment_a(seed=0, n=48, steps=220, tail=60, noise=0.7)
        self.assertGreater(r.improvement, 0.0)

    def test_experiment_b_improves_with_gate(self):
        r = run_experiment_b(seed=0, n=48, steps1=140, steps2=200, tail=80, noise=0.7)
        self.assertGreater(r.improvement, 0.0)
        self.assertGreater(r.gate_triggers, 0)


if __name__ == "__main__":
    unittest.main()


