import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from src.core.experiment import run_experiment_a, run_experiment_b, sanity_energy
from src.core.designer import DesignOptimizer
from src.envs.simple_combat import GridCombatEnv


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

    def test_grid_env_obstacles_respect_spawn_columns(self):
        design = {
            "n_factions": 2,
            "width": 12,
            "height": 12,
            "obstacle_density": 0.25,
            "obstacle_pattern": 1,
            "max_steps": 30,
            "no_attack_limit": 15,
            "shaping_scale": 0.0,
            # 작은 테스트용 편성(king은 자동 1)
            "p0_unit0_units": 1,
            "p0_unit1_units": 0,
            "p0_unit2_units": 0,
            "p0_unit3_units": 0,
            "p0_unit4_units": 0,
            "p1_unit0_units": 1,
            "p1_unit1_units": 0,
            "p1_unit2_units": 0,
            "p1_unit3_units": 0,
            "p1_unit4_units": 0,
        }
        env = GridCombatEnv(design, seed=0, factions=(0, 1))
        env.reset(first_turn=0)
        banned_cols = env._spawn_columns()
        for (x, _y) in getattr(env, "obstacles", set()):
            self.assertNotIn(int(x), banned_cols)

    def test_grid_env_step_smoke(self):
        design = {
            "n_factions": 2,
            "width": 12,
            "height": 12,
            "obstacle_density": 0.0,
            "obstacle_pattern": 0,
            "max_steps": 20,
            "no_attack_limit": 10,
            "shaping_scale": 0.0,
            # 작은 테스트용 편성(king은 자동 1)
            "p0_unit0_units": 1,
            "p0_unit1_units": 0,
            "p0_unit2_units": 0,
            "p0_unit3_units": 0,
            "p0_unit4_units": 0,
            "p1_unit0_units": 1,
            "p1_unit1_units": 0,
            "p1_unit2_units": 0,
            "p1_unit3_units": 0,
            "p1_unit4_units": 0,
        }
        env = GridCombatEnv(design, seed=0, factions=(0, 1))
        obs = env.reset(first_turn=0)
        self.assertEqual(len(obs), 2)
        self.assertEqual(len(obs[0]), int(env.n_units[0]))
        self.assertEqual(len(obs[1]), int(env.n_units[1]))
        self.assertTrue(all(isinstance(u, torch.Tensor) and int(u.numel()) == 12 for u in obs[0]))
        self.assertTrue(all(isinstance(u, torch.Tensor) and int(u.numel()) == 12 for u in obs[1]))

        done = False
        steps = 0
        # 항상 "현재 턴 + 유닛0 + action0"만 넣어도 크래시 없이 종료되어야 한다.
        while not done and steps < 40:
            side = int(getattr(env, "side_to_act", 0))
            obs, rewards, done, info = env.step((side, 0, 0))
            self.assertEqual(len(rewards), 2)
            self.assertIn("attack_distances", info)
            steps += 1

        self.assertTrue(done)
        self.assertIn(info["winner"], (0, 1, -1))

    def test_design_optimizer_clamp_basic(self):
        initial = {
            "n_factions": 2,
            "width": 11.1,
            "height": 23.9,
            "obstacle_density": 1.0,
            "obstacle_pattern": 10,
            "p0_unit0_units": -3,
            "p1_unit0_units": 99,
        }
        opt = DesignOptimizer(
            initial,
            target_dist_mean=3.0,
            n_samples=2,
            train_episodes=1,
            eval_episodes=1,
            use_parallel=False,
            verbose=False,
        )
        clamped = opt._clamp_design(initial)
        self.assertGreaterEqual(int(clamped["width"]), 10)
        self.assertLessEqual(int(clamped["width"]), 24)
        self.assertEqual(int(clamped["width"]) % 2, 0)
        self.assertGreaterEqual(int(clamped["height"]), 10)
        self.assertLessEqual(int(clamped["height"]), 24)
        self.assertEqual(int(clamped["height"]) % 2, 0)
        self.assertGreaterEqual(float(clamped["obstacle_density"]), 0.0)
        self.assertLessEqual(float(clamped["obstacle_density"]), 0.35)
        self.assertGreaterEqual(int(clamped["obstacle_pattern"]), 0)
        self.assertLessEqual(int(clamped["obstacle_pattern"]), 3)
        self.assertGreaterEqual(int(clamped["p0_unit0_units"]), 0)
        self.assertLessEqual(int(clamped["p0_unit0_units"]), 8)
        self.assertGreaterEqual(int(clamped["p1_unit0_units"]), 0)
        self.assertLessEqual(int(clamped["p1_unit0_units"]), 5)


if __name__ == "__main__":
    unittest.main()


