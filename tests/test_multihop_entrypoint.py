import ast
import unittest
from pathlib import Path

import numpy as np

from envs.env_continuous_multihop import ContinuousMultihopEnv


ROOT = Path(__file__).resolve().parents[1]


class MultihopEntrypointTests(unittest.TestCase):
    def test_wrapper_spaces_and_shapes(self):
        env = ContinuousMultihopEnv()
        self.assertEqual(len(env.action_space), 4)
        self.assertEqual(env.action_space[0].shape, (2,))
        self.assertEqual(env.observation_space[0].shape, (90,))
        self.assertEqual(env.share_observation_space[0].shape, (360,))

        obs = env.reset()
        self.assertEqual(obs.shape, (4, 90))
        obs, rewards, dones, info = env.step(np.zeros((4, 2)))

        self.assertEqual(obs.shape, (4, 90))
        self.assertEqual(rewards.shape, (4, 1))
        self.assertEqual(dones.shape, (4,))
        self.assertIn("system_time", info)

    def test_sixty_step_episode(self):
        env = ContinuousMultihopEnv()
        env.seed(1)
        obs = env.reset()
        for _step in range(60):
            obs, rewards, dones, info = env.step(np.zeros((4, 2)))
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertTrue(np.all(np.isfinite(rewards)))
            self.assertTrue(
                np.all(np.isin(env.env.offloading_decisions, [0, 1]))
            )
            self.assertEqual(info["avg_uav_comp_energy"], 0.0)
        self.assertTrue(np.all(dones))

    def test_wrapper_rejects_nonfinite_or_malformed_actions(self):
        env = ContinuousMultihopEnv()
        env.seed(2)
        env.reset()

        with self.assertRaises(ValueError):
            env.step(np.zeros((3, 2)))
        actions = np.zeros((4, 2))
        actions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            env.step(actions)

        observations, rewards, _dones, _info = env.step(
            np.full((4, 2), 1.5)
        )
        self.assertTrue(np.all(np.isfinite(observations)))
        self.assertTrue(np.all(np.isfinite(rewards)))

    def test_environment_seeds_are_independent(self):
        first = ContinuousMultihopEnv()
        second = ContinuousMultihopEnv()
        reference = ContinuousMultihopEnv()
        first.seed(1)
        second.seed(1001)
        first_obs = first.reset()
        reference.seed(1)
        reference_obs = reference.reset()

        np.testing.assert_allclose(first_obs, reference_obs, rtol=0, atol=0)
        actions = np.zeros((4, 2))
        first_next, first_rewards, _first_dones, _first_info = first.step(
            actions
        )
        reference_next, reference_rewards, _reference_dones, _reference_info = (
            reference.step(actions)
        )
        np.testing.assert_allclose(first_next, reference_next, rtol=0, atol=0)
        np.testing.assert_allclose(
            first_rewards, reference_rewards, rtol=0, atol=0
        )

    def test_training_arguments_cannot_break_fixed_invariants(self):
        from config import get_config
        from train.train_multihop import _validate_fixed_args, parse_args

        parser = get_config()
        valid = parse_args([], parser)
        _validate_fixed_args(valid)
        self.assertFalse(valid.use_linear_lr_decay)
        with_decay = parse_args(
            ["--use_linear_lr_decay"], get_config()
        )
        self.assertTrue(with_decay.use_linear_lr_decay)
        invalid_cases = (
            ["--env_name", "MyEnv"],
            ["--episode_length", "30"],
            ["--num_agents", "3"],
            ["--share_policy"],
        )
        for command in invalid_cases:
            with self.subTest(command=command):
                args = parse_args(command, get_config())
                with self.assertRaises(ValueError):
                    _validate_fixed_args(args)

    def test_dedicated_runner_persists_multihop_metrics(self):
        from runner.shared.env_runner_multihop import MULTIHOP_METRIC_KEYS

        required = {
            "avg_uav_relay_energy",
            "bs_offloading_ratio",
            "route_availability_ratio",
            "avg_hop_count",
            "max_hop_count",
            "potential_passes",
        }
        self.assertTrue(required.issubset(MULTIHOP_METRIC_KEYS))

    def test_training_entry_imports_only_multihop_wrapper(self):
        source = (ROOT / "train" / "train_multihop.py").read_text()
        ast.parse(source)
        self.assertIn("ContinuousMultihopEnv", source)
        self.assertIn('env_name="MultihopUAVBS"', source)
        self.assertIn("use_linear_lr_decay=False", source)
        self.assertIn("env_runner_multihop", source)
        self.assertNotIn("ContinuousActionEnv", source)


if __name__ == "__main__":
    unittest.main()
