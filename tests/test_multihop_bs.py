import unittest

import numpy as np

from envs.envs_multihop_bs import EnvCore
from envs.envs_202509 import EnvCore as OriginalEnvCore


class MultihopTopologyTests(unittest.TestCase):
    def setUp(self):
        self.env = EnvCore()
        self.env.seed(7)
        self.env.reset()

    def test_default_parameters_and_initial_backhaul(self):
        self.assertTrue(np.allclose(self.env.bs["position"], [500.0, 0.0]))
        self.assertEqual(self.env.base.bs_resource, 100 * self.env.base.GHz)
        self.assertEqual(self.env.base.carrier_frequency_uav, 5e9)
        self.assertEqual(self.env.base.uav_backhaul_carrier_frequency, 2e9)
        self.assertEqual(self.env.num_usvs, 20)
        self.assertEqual(self.env.obs_dim, 90)

        graph = self.env._build_backhaul_graph()
        paths = [
            self.env._shortest_uav_path_to_bs(i, graph)
            for i in range(self.env.num_uavs)
        ]

        self.assertTrue(all(path is not None for path in paths))
        self.assertEqual(paths[0][-1], self.env.bs_node)
        self.assertGreaterEqual(len(paths[2]), 3)

    def test_access_and_backhaul_carrier_frequencies_are_isolated(self):
        args = (
            np.array([100.0, 100.0]),
            self.env.base.H_UAV,
            np.array([400.0, 100.0]),
            self.env.base.H_UAV,
        )
        _, baseline_rate = self.env._air_link(*args)

        self.env.base.carrier_frequency_uav = 9e9
        _, access_changed_rate = self.env._air_link(*args)
        self.assertEqual(access_changed_rate, baseline_rate)

        self.env.base.uav_backhaul_carrier_frequency = 3e9
        _, backhaul_changed_rate = self.env._air_link(*args)
        self.assertNotEqual(backhaul_changed_rate, baseline_rate)

    def test_no_direct_usv_bs_route(self):
        self.env.usvs[0]["position"] = np.array([500.0, 10.0])
        for uav in self.env.uavs:
            uav["position"] = np.array([900.0, 900.0])

        route = self.env._build_routes()[0]

        self.assertIsNone(route)

    def test_route_has_access_and_final_bs_hop(self):
        self.env.usvs[0]["position"] = np.array([110.0, 110.0])

        route = self.env._build_routes()[0]

        self.assertIsNotNone(route)
        self.assertEqual(route["access_uav"], 0)
        self.assertEqual(
            route["backhaul_hops"][-1]["rx_node"], self.env.bs_node
        )
        self.assertTrue(
            all(hop["rate"] > 0 for hop in route["backhaul_hops"])
        )


class MultihopAllocationTests(unittest.TestCase):
    def setUp(self):
        self.env = EnvCore()
        self.env.seed(13)
        self.env.reset()
        positions = ([300.0, 300.0], [700.0, 300.0],
                     [300.0, 700.0], [700.0, 700.0])
        for uav, position in zip(self.env.uavs, positions):
            uav["position"] = np.array(position)
        access_positions = ([310.0, 310.0], [690.0, 310.0],
                            [310.0, 690.0], [690.0, 690.0])
        for usv, position in zip(self.env.usvs[:4], access_positions):
            usv["position"] = np.array(position)

    def test_resource_shares_respect_every_pool(self):
        routes = self.env._build_routes()
        decisions = np.zeros(self.env.num_usvs, dtype=int)
        decisions[:4] = 1

        allocation = self.env._allocate_resources(decisions, routes)

        for access_uav in range(self.env.num_uavs):
            selected = [
                k for k, route in enumerate(routes)
                if decisions[k] == 1
                and route is not None
                and route["access_uav"] == access_uav
            ]
            if selected:
                self.assertAlmostEqual(
                    float(np.sum(allocation["access_shares"][selected])),
                    1.0,
                    places=7,
                )
        self.assertAlmostEqual(
            sum(allocation["backhaul_shares"].values()), 1.0, places=7
        )
        self.assertAlmostEqual(
            float(np.sum(allocation["bs_cpu_shares"])), 1.0, places=7
        )

    def test_potential_game_returns_binary_local_minimum(self):
        routes = self.env._build_routes()

        decisions = self.env._run_potential_game(routes)

        self.assertTrue(np.all(np.isin(decisions, [0, 1])))
        final_value = self.env._evaluate_profile(decisions, routes)["potential"]
        self.assertTrue(np.isfinite(final_value))
        for k in range(self.env.num_usvs):
            alternative = decisions.copy()
            alternative[k] = 1 - alternative[k]
            if alternative[k] == 1 and routes[k] is None:
                continue
            alternative_value = self.env._evaluate_profile(
                alternative, routes
            )["potential"]
            self.assertGreaterEqual(
                alternative_value + self.env.base.potential_tolerance,
                final_value,
            )

    def test_potential_game_does_not_silently_return_at_pass_cap(self):
        routes = self.env._build_routes()
        self.env.base.potential_max_passes = 0

        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            self.env._run_potential_game(routes)

    def test_extra_relay_hop_increases_delay_and_attributes_energy(self):
        decisions = np.zeros(self.env.num_usvs, dtype=int)
        decisions[0] = 1
        common_route = {
            "access_uav": 0,
            "access_rate": 100e6,
            "access_distance": 100.0,
        }
        direct_routes = [None] * self.env.num_usvs
        direct_routes[0] = {
            **common_route,
            "backhaul_hops": [
                {
                    "tx_uav": 0,
                    "rx_node": self.env.bs_node,
                    "rate": 100e6,
                    "distance": 300.0,
                }
            ],
        }
        relay_routes = [None] * self.env.num_usvs
        relay_routes[0] = {
            **common_route,
            "backhaul_hops": [
                {
                    "tx_uav": 0,
                    "rx_node": 1,
                    "rate": 100e6,
                    "distance": 300.0,
                },
                {
                    "tx_uav": 1,
                    "rx_node": self.env.bs_node,
                    "rate": 100e6,
                    "distance": 300.0,
                },
            ],
        }

        direct = self.env._evaluate_profile(decisions, direct_routes)
        relay = self.env._evaluate_profile(decisions, relay_routes)

        self.assertGreater(relay["task_times"][0], direct["task_times"][0])
        self.assertGreater(direct["uav_relay_energies"][0], 0.0)
        self.assertEqual(direct["uav_relay_energies"][1], 0.0)
        self.assertGreater(relay["uav_relay_energies"][0], 0.0)
        self.assertGreater(relay["uav_relay_energies"][1], 0.0)
        self.assertTrue(np.all(relay["uav_relay_energies"][2:] == 0.0))


class MultihopEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = EnvCore()
        self.env.seed(17)

    def test_reset_and_step_shapes_and_metrics(self):
        obs = self.env.reset()
        self.assertEqual(len(obs), 4)
        self.assertTrue(all(item.shape == (90,) for item in obs))
        self.assertTrue(all(np.all(np.isfinite(item)) for item in obs))

        next_obs, rewards, dones, info = self.env.step(
            np.zeros((4, 2), dtype=float)
        )

        self.assertEqual(np.asarray(next_obs).shape, (4, 90))
        self.assertEqual(rewards.shape, (4, 1))
        self.assertEqual(len(dones), 4)
        self.assertTrue(np.all(np.isfinite(rewards)))
        self.assertIn("avg_uav_relay_energy", info)
        self.assertIn("bs_offloading_ratio", info)
        self.assertIn("avg_hop_count", info)
        self.assertTrue(0.0 <= info["route_availability_ratio"] <= 100.0)
        self.assertEqual(info["avg_uav_comp_energy"], 0.0)

    def test_episode_terminates_after_sixty_steps(self):
        self.env.reset()
        dones = [False] * 4
        for _ in range(60):
            _obs, _rewards, dones, _info = self.env.step(
                np.zeros((4, 2), dtype=float)
            )
        self.assertTrue(all(dones))

    def test_reward_matches_existing_arithmetic(self):
        self.env.reset()
        self.env.system_time = 3.0
        self.env.task_completion_rate = 80.0
        for idx, uav in enumerate(self.env.uavs):
            uav["position"] = np.array([200.0 + 200.0 * idx, 500.0])
            uav["trajectory"] = [
                uav["position"].copy(),
                uav["position"].copy(),
            ]
        service = np.zeros(4)
        flight = np.full(4, 700.0)

        actual = self.env.calculate_rewards(
            np.zeros(self.env.num_usvs), 80.0, service, flight
        )

        completion_reward = 30.0 * (
            80.0 / self.env.base.completion_threshold
        )
        energy_penalty = -20.0 * (
            min(2.0, 700.0 / self.env.base.energy_threshold) - 1.0
        )
        for idx in range(4):
            coverage = self.env._calculate_coverage_reward(idx)
            stability = self.env._calculate_stability_reward(idx)
            expected = (
                self.env.base.w_delay * -3.0
                + self.env.base.w_energy * energy_penalty
                + self.env.base.w_completion * completion_reward
                + coverage
                + stability
            )
            self.assertAlmostEqual(actual[idx], expected, places=7)

    def test_reward_is_equal_to_original_for_equal_inputs(self):
        self.env.reset()
        original = OriginalEnvCore()
        original.usvs = self.env.usvs
        original.uavs = self.env.uavs
        # The original experiment currently uses a 40-USV sweep, while this
        # comparison experiment intentionally fixes 20 USVs.
        original.num_usvs = self.env.num_usvs
        original.system_time = self.env.system_time = 2.75
        original.task_completion_rate = self.env.task_completion_rate = 82.0
        for idx, uav in enumerate(self.env.uavs):
            uav["position"] = np.array([150.0 + 220.0 * idx, 450.0])
            uav["trajectory"] = [
                uav["position"].copy(),
                uav["position"].copy(),
            ]
        service = np.array([0.4, 0.8, 0.0, 1.2])
        flight = np.array([590.0, 610.0, 650.0, 720.0])

        expected = original.calculate_rewards(
            np.zeros(self.env.num_usvs), 82.0, service, flight
        )
        actual = self.env.calculate_rewards(
            np.zeros(self.env.num_usvs), 82.0, service, flight
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
