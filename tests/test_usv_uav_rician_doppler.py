import unittest
from unittest import mock

import numpy as np

from envs.Init_player import init_uav, init_usv
from envs.base import base
from envs.common import common
from envs.envs_202509 import EnvCore as OriginalEnvCore


class UsvUavChannelParameterTests(unittest.TestCase):
    def test_access_channel_parameters(self):
        config = base()

        self.assertEqual(config.carrier_frequency_uav, 5e9)
        self.assertEqual(config.rician_K_factor_uav_db, 15.0)
        self.assertAlmostEqual(
            config.rician_K_factor_uav, 31.622776601683793
        )
        self.assertEqual(config.subcarrier_spacing, 30000)
        self.assertAlmostEqual(config.ofdm_symbol_duration, 1.0 / 30000)
        self.assertEqual(config.doppler_compensation_ratio, 0.9)
        self.assertAlmostEqual(config.doppler_residual_ratio, 0.1)
        self.assertEqual(config.run_slot, 1.0)

    def test_initial_velocity_vectors_match_motion_state(self):
        config = base()
        config.num_usv = 3
        config.num_uav = 2
        np.random.seed(29)

        usvs = init_usv(config)
        uavs = init_uav(config)

        for usv in usvs:
            expected = usv["velocity"] * np.array(
                [np.cos(usv["direction"]), np.sin(usv["direction"])]
            )
            np.testing.assert_allclose(usv["velocity_vector"], expected)
        for uav in uavs:
            np.testing.assert_array_equal(uav["velocity_vector"], np.zeros(2))


class UsvUavRicianDopplerTests(unittest.TestCase):
    def setUp(self):
        self.config = base()
        self.utils = common(self.config)

    def test_rician_power_is_normalized(self):
        np.random.seed(1234)
        samples = np.array(
            [
                self.utils.calculate_usv_to_uav_rician_power(250.0)
                for _ in range(20000)
            ]
        )

        self.assertAlmostEqual(float(np.mean(samples)), 1.0, delta=0.03)
        self.assertTrue(np.all(samples >= 0.0))

    def test_equal_velocity_has_zero_doppler_and_unit_kappa(self):
        velocity = np.array([4.0, -2.0])
        usv = {
            "position": np.array([100.0, 50.0]),
            "velocity_vector": velocity.copy(),
        }
        uav = {
            "position": np.array([0.0, 0.0]),
            "velocity_vector": velocity.copy(),
        }

        doppler, residual, kappa = self.utils.calculate_usv_to_uav_doppler(
            usv, uav
        )

        self.assertEqual(doppler, 0.0)
        self.assertEqual(residual, 0.0)
        self.assertEqual(kappa, 1.0)

    def test_radial_speed_increases_doppler_and_does_not_increase_kappa(self):
        usv = {
            "position": np.array([100.0, 0.0]),
            "velocity_vector": np.array([5.0, 0.0]),
        }
        uav = {
            "position": np.array([0.0, 0.0]),
            "velocity_vector": np.zeros(2),
        }
        slow = self.utils.calculate_usv_to_uav_doppler(usv, uav)
        usv["velocity_vector"] = np.array([10.0, 0.0])
        fast = self.utils.calculate_usv_to_uav_doppler(usv, uav)

        self.assertGreater(abs(fast[0]), abs(slow[0]))
        self.assertLessEqual(fast[2], slow[2])

    def test_maximum_expected_radial_speed_keeps_kappa_near_one(self):
        usv = {
            "position": np.array([10000.0, 0.0]),
            "velocity_vector": np.array([33.0, 0.0]),
        }
        uav = {
            "position": np.array([0.0, 0.0]),
            "velocity_vector": np.zeros(2),
        }

        _, _, kappa = self.utils.calculate_usv_to_uav_doppler(usv, uav)

        self.assertGreater(kappa, 0.999)
        self.assertLessEqual(kappa, 1.0)


class DopplerAwareRateTests(unittest.TestCase):
    def setUp(self):
        self.config = base()
        self.utils = common(self.config)

    def test_unit_kappa_matches_legacy_shannon_rate(self):
        power = 0.2
        gain = 1e-10
        bandwidth = 20e6
        expected = bandwidth * np.log2(
            1.0 + power * gain / self.config.noise_power_density
        )

        actual = self.utils.calculate_rate_bps(
            power, gain, bandwidth, doppler_coefficient=1.0
        )

        self.assertAlmostEqual(actual, expected, places=7)

    def test_lower_kappa_cannot_increase_rate(self):
        rates = [
            self.utils.calculate_rate_bps(0.2, 1e-10, 20e6, kappa)
            for kappa in (1.0, 0.9, 0.5, 0.0)
        ]

        self.assertTrue(all(np.isfinite(rates)))
        self.assertTrue(all(rate >= 0.0 for rate in rates))
        self.assertTrue(
            all(left >= right for left, right in zip(rates, rates[1:]))
        )


class ChannelSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.config = base()
        self.config.num_usv = 3
        self.config.num_uav = 2
        self.utils = common(self.config)
        np.random.seed(41)
        self.usvs = init_usv(self.config)
        self.uavs = init_uav(self.config)

    def test_snapshot_shapes_and_stable_reads(self):
        snapshot = self.utils.build_usv_to_uav_channel_snapshot(
            self.usvs, self.uavs
        )

        for key in (
            "gain",
            "rician_power",
            "doppler_hz",
            "residual_doppler_hz",
            "kappa",
        ):
            self.assertEqual(snapshot[key].shape, (3, 2))
        first_read = snapshot["gain"][1, 1]
        self.assertEqual(first_read, snapshot["gain"][1, 1])
        self.assertTrue(np.all(snapshot["gain"] >= 0.0))
        self.assertTrue(np.all((snapshot["kappa"] >= 0.0)))
        self.assertTrue(np.all((snapshot["kappa"] <= 1.0)))

    def test_new_slot_can_sample_new_rician_power(self):
        first = self.utils.build_usv_to_uav_channel_snapshot(
            self.usvs, self.uavs
        )
        second = self.utils.build_usv_to_uav_channel_snapshot(
            self.usvs, self.uavs
        )

        self.assertFalse(
            np.array_equal(first["rician_power"], second["rician_power"])
        )

    def test_fixed_seed_reproduces_snapshot(self):
        np.random.seed(2026)
        first = self.utils.build_usv_to_uav_channel_snapshot(
            self.usvs, self.uavs
        )
        np.random.seed(2026)
        second = self.utils.build_usv_to_uav_channel_snapshot(
            self.usvs, self.uavs
        )

        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_consumers_reuse_snapshot_without_resampling(self):
        snapshot = self.utils.build_usv_to_uav_channel_snapshot(
            self.usvs, self.uavs
        )
        original_sampler = self.utils.calculate_usv_to_uav_rician_power

        def fail_if_resampled(_distance):
            raise AssertionError("consumer resampled the access channel")

        self.utils.calculate_usv_to_uav_rician_power = fail_if_resampled
        try:
            pc, pb = self.utils.get_uav_pc_pb(
                self.usvs,
                np.array([0, 1]),
                self.uavs[0],
                0,
                self.config,
                snapshot,
            )
            rate = self.utils.calculate_rate_bps(
                self.usvs[0]["power"],
                snapshot["gain"][0, 0],
                self.uavs[0]["bandwith"],
                snapshot["kappa"][0, 0],
            )
        finally:
            self.utils.calculate_usv_to_uav_rician_power = original_sampler

        self.assertTrue(np.all(np.isfinite(pc)))
        self.assertTrue(np.all(np.isfinite(pb)))
        self.assertTrue(np.isfinite(rate))

    def test_original_environment_samples_once_per_link_per_slot(self):
        original_sampler = common.calculate_usv_to_uav_rician_power
        sampled_distances = []

        def counted_sampler(instance, distance):
            sampled_distances.append(distance)
            return original_sampler(instance, distance)

        np.random.seed(77)
        with mock.patch.object(
            common,
            "calculate_usv_to_uav_rician_power",
            new=counted_sampler,
        ):
            env = OriginalEnvCore()
            env.reset()
            expected = env.num_usvs * env.num_uavs
            self.assertEqual(len(sampled_distances), expected)

            sampled_distances.clear()
            env.step(np.zeros((env.num_uavs, 2), dtype=float))
            self.assertEqual(len(sampled_distances), expected)


if __name__ == "__main__":
    unittest.main()
