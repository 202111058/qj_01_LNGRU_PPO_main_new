"""Standalone multihop UAV relay environment for terrestrial BS offloading."""

import heapq

import numpy as np

from .base import base
from .Init_player import init_uav, init_usv
from .common import common


class MultihopBase(base):
    """Parameters used only by the multihop UAV-BS experiment."""

    def __init__(self):
        super().__init__()
        # Keep the comparison experiment independent from the original
        # experiment's device-count sweep.  The multihop design and its
        # 90-dimensional observation both assume 4 UAVs and 20 USVs.
        self.num_uav = 4
        self.num_usv = 20
        self.episode_length = 60
        self.bs_position = np.array([500.0, 0.0], dtype=float)
        self.H_BS = 30.0
        self.bs_resource = 100 * self.GHz
        self.uav_uav_coverage = 850.0
        self.uav_bs_coverage = 450.0
        self.uav_relay_power = 1.0
        self.uav_access_bandwidth = 20 * self.MHz
        self.uav_backhaul_bandwidth = 20 * self.MHz
        self.potential_max_passes = 50
        self.potential_tolerance = 1e-9


def init_bs(config):
    """Create the fixed terrestrial BS used by the multihop environment."""

    return {
        "position": config.bs_position.copy(),
        "height": config.H_BS,
        "resource": config.bs_resource,
    }


class EnvCore:
    """Core multihop environment; UAVs relay but never compute tasks."""

    def __init__(self):
        self.base = MultihopBase()
        self.common = common(self.base)
        self.num_usvs = self.base.num_usv
        self.num_uavs = self.base.num_uav
        self.bs_node = self.num_uavs
        self.obs_dim = (
            2 + 2 * (self.num_uavs - 1) + 2 + 4 * self.num_usvs
        )
        self.action_dim = 2
        self.usvs = []
        self.uavs = []
        self.bs = None
        self.offloading_decisions = np.zeros(self.num_usvs, dtype=int)
        self.current_step = 0
        self.last_potential_passes = 0
        self.last_potential_converged = False
        self._rng = np.random.RandomState()
        self.system_time = 0.0
        self.total_energy = 0.0
        self.avg_usv_energy = 0.0
        self.avg_uav_energy = 0.0
        self.task_completion_rate = 0.0

    def seed(self, seed):
        """Seed this environment without changing another environment's RNG."""

        self._rng.seed(seed)
        return [seed]

    def _run_with_rng(self, operation, *args):
        """Run legacy NumPy-based helpers with an environment-owned stream."""

        global_state = np.random.get_state()
        np.random.set_state(self._rng.get_state())
        try:
            return operation(*args)
        finally:
            self._rng.set_state(np.random.get_state())
            np.random.set_state(global_state)

    def reset(self):
        return self._run_with_rng(self._reset)

    def _reset(self):
        """Initialize all nodes and return four 90-dimensional observations."""

        self.current_step = 0
        self.usvs = init_usv(self.base)
        self.uavs = init_uav(self.base)
        self.bs = init_bs(self.base)
        self.offloading_decisions.fill(0)
        self.system_time = 0.0
        self.total_energy = 0.0
        self.avg_usv_energy = 0.0
        self.avg_uav_energy = 0.0
        self.task_completion_rate = 0.0
        self.last_potential_passes = 0
        self.last_potential_converged = False
        return self._get_observation()

    @staticmethod
    def _normalize(value, minimum, maximum):
        span = maximum - minimum
        if span == 0:
            return 0.5
        return (value - minimum) / span

    def _get_observation(self):
        """Construct topology-aware observations in stable UAV index order."""

        observations = []
        for own_idx, uav in enumerate(self.uavs):
            obs = [
                self._normalize(
                    uav["position"][0], self.base.field_X[0], self.base.field_X[1]
                ),
                self._normalize(
                    uav["position"][1], self.base.field_Y[0], self.base.field_Y[1]
                ),
            ]
            for other_idx, other_uav in enumerate(self.uavs):
                if other_idx == own_idx:
                    continue
                obs.extend(
                    [
                        self._normalize(
                            other_uav["position"][0],
                            self.base.field_X[0],
                            self.base.field_X[1],
                        ),
                        self._normalize(
                            other_uav["position"][1],
                            self.base.field_Y[0],
                            self.base.field_Y[1],
                        ),
                    ]
                )
            obs.extend(
                [
                    self._normalize(
                        self.bs["position"][0],
                        self.base.field_X[0],
                        self.base.field_X[1],
                    ),
                    self._normalize(
                        self.bs["position"][1],
                        self.base.field_Y[0],
                        self.base.field_Y[1],
                    ),
                ]
            )
            for usv in self.usvs:
                obs.extend(
                    [
                        self._normalize(
                            usv["position"][0],
                            self.base.field_X[0],
                            self.base.field_X[1],
                        ),
                        self._normalize(
                            usv["position"][1],
                            self.base.field_Y[0],
                            self.base.field_Y[1],
                        ),
                        self._normalize(
                            usv["task_size"],
                            self.base.task_size_min,
                            self.base.task_size_max,
                        ),
                        self._normalize(
                            usv["task_resource"],
                            self.base.task_resources_min,
                            self.base.task_resources_max,
                        ),
                    ]
                )
            observation = np.asarray(obs, dtype=np.float32)
            if observation.shape != (self.obs_dim,):
                raise ValueError(
                    f"Expected observation shape {(self.obs_dim,)}, "
                    f"received {observation.shape}"
                )
            observations.append(observation)
        return observations

    def _air_link(self, pos_a, height_a, pos_b, height_b):
        """Return 3-D distance and LoS free-space relay rate."""

        horizontal = float(
            np.linalg.norm(np.asarray(pos_a, dtype=float) - np.asarray(pos_b, dtype=float))
        )
        distance = float(np.hypot(horizontal, height_a - height_b))
        if distance <= 0:
            return 0.0, float("inf")
        path_loss_db = 20 * np.log10(
            4
            * np.pi
            * distance
            * self.base.carrier_frequency_uav
            / self.base.light_speed
        )
        gain = 10 ** (-path_loss_db / 10)
        rate = self.common.calculate_rate_bps(
            self.base.uav_relay_power,
            gain,
            self.base.uav_backhaul_bandwidth,
        )
        return distance, float(rate)

    def _build_backhaul_graph(self):
        """Build directed UAV-UAV and UAV-BS edges for the current slot."""

        graph = {node: [] for node in range(self.num_uavs + 1)}
        for src in range(self.num_uavs):
            for dst in range(self.num_uavs):
                if src == dst:
                    continue
                distance, rate = self._air_link(
                    self.uavs[src]["position"],
                    self.base.H_UAV,
                    self.uavs[dst]["position"],
                    self.base.H_UAV,
                )
                if distance <= self.base.uav_uav_coverage:
                    graph[src].append((dst, rate, distance))

            distance, rate = self._air_link(
                self.uavs[src]["position"],
                self.base.H_UAV,
                self.bs["position"],
                self.bs["height"],
            )
            if distance <= self.base.uav_bs_coverage:
                graph[src].append((self.bs_node, rate, distance))
        return graph

    def _shortest_uav_path_to_bs(self, start_uav, graph):
        """Find the minimum raw per-bit-delay UAV path to the fixed BS."""

        queue = [(0.0, start_uav, [])]
        visited = set()
        while queue:
            cost, node, prefix = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            path = prefix + [node]
            if node == self.bs_node:
                return path
            for next_node, rate, _distance in graph[node]:
                if next_node not in visited and rate > 0:
                    heapq.heappush(
                        queue, (cost + 1.0 / rate, next_node, path)
                    )
        return None

    @staticmethod
    def _graph_edge(graph, src, dst):
        for next_node, rate, distance in graph[src]:
            if next_node == dst:
                return rate, distance
        raise ValueError(f"Missing graph edge {src}->{dst}")

    def _build_routes(self):
        """Choose one minimum raw-delay multihop route for every USV."""

        graph = self._build_backhaul_graph()
        uav_paths = [
            self._shortest_uav_path_to_bs(uav_idx, graph)
            for uav_idx in range(self.num_uavs)
        ]
        routes = []
        for usv in self.usvs:
            candidates = []
            for access_uav, path in enumerate(uav_paths):
                if path is None:
                    continue
                horizontal = float(
                    np.linalg.norm(
                        usv["position"] - self.uavs[access_uav]["position"]
                    )
                )
                if horizontal > self.base.uav_coverage:
                    continue
                access_distance = float(
                    np.hypot(horizontal, self.base.H_UAV)
                )
                gain = self.common.calculate_usv_to_uav_channel_power_gain(
                    usv["position"], self.uavs[access_uav]["position"]
                )
                access_rate = float(
                    self.common.calculate_rate_bps(
                        usv["power"], gain, self.base.uav_access_bandwidth
                    )
                )
                if access_rate <= 0:
                    continue
                backhaul_hops = []
                raw_cost = 1.0 / access_rate
                for tx_uav, rx_node in zip(path[:-1], path[1:]):
                    rate, distance = self._graph_edge(
                        graph, tx_uav, rx_node
                    )
                    raw_cost += 1.0 / rate
                    backhaul_hops.append(
                        {
                            "tx_uav": tx_uav,
                            "rx_node": rx_node,
                            "rate": float(rate),
                            "distance": float(distance),
                        }
                    )
                candidates.append(
                    (
                        raw_cost,
                        {
                            "access_uav": access_uav,
                            "access_rate": access_rate,
                            "access_distance": access_distance,
                            "backhaul_hops": backhaul_hops,
                        },
                    )
                )
            routes.append(min(candidates, key=lambda item: item[0])[1] if candidates else None)
        return routes

    def _allocate_resources(self, decisions, routes):
        """Allocate access bandwidth, shared backhaul, and BS CPU."""

        decisions = np.asarray(decisions, dtype=int)
        access_shares = np.zeros(self.num_usvs, dtype=float)
        bs_cpu_shares = np.zeros(self.num_usvs, dtype=float)
        backhaul_shares = {}
        selected = [
            k
            for k in range(self.num_usvs)
            if decisions[k] == 1 and routes[k] is not None
        ]

        for access_uav in range(self.num_uavs):
            group = [
                k
                for k in selected
                if routes[k]["access_uav"] == access_uav
            ]
            if not group:
                continue
            weights = np.array(
                [
                    np.sqrt(
                        self.usvs[k]["task_size"]
                        / routes[k]["access_rate"]
                    )
                    for k in group
                ],
                dtype=float,
            )
            denominator = float(np.sum(weights))
            if denominator > 1e-12:
                access_shares[group] = weights / denominator

        if selected:
            cpu_weights = np.array(
                [
                    np.sqrt(
                        self.usvs[k]["task_size"]
                        * self.usvs[k]["task_resource"]
                        / self.bs["resource"]
                    )
                    for k in selected
                ],
                dtype=float,
            )
            denominator = float(np.sum(cpu_weights))
            if denominator > 1e-12:
                bs_cpu_shares[selected] = cpu_weights / denominator

        operations = []
        weights = []
        for k in selected:
            for hop_idx, hop in enumerate(routes[k]["backhaul_hops"]):
                operations.append((k, hop_idx))
                weights.append(
                    np.sqrt(self.usvs[k]["task_size"] / hop["rate"])
                )
        denominator = float(np.sum(weights))
        if denominator > 1e-12:
            for operation, weight in zip(operations, weights):
                backhaul_shares[operation] = float(weight / denominator)

        return {
            "access_shares": access_shares,
            "bs_cpu_shares": bs_cpu_shares,
            "backhaul_shares": backhaul_shares,
        }

    def _evaluate_profile(self, decisions, routes):
        """Evaluate delay and energy for one complete offloading profile."""

        decisions = np.asarray(decisions, dtype=int)
        allocation = self._allocate_resources(decisions, routes)
        task_times = np.zeros(self.num_usvs, dtype=float)
        usv_energies = np.zeros(self.num_usvs, dtype=float)
        uav_relay_energies = np.zeros(self.num_uavs, dtype=float)

        for k, usv in enumerate(self.usvs):
            task_size = float(usv["task_size"])
            task_resource = float(usv["task_resource"])
            if decisions[k] == 0:
                local_time = self.common.calculate_computation_time(
                    task_size, task_resource, usv["resource"]
                )
                task_times[k] = local_time
                usv_energies[k] = 0.1 * local_time
                continue

            route = routes[k]
            access_share = allocation["access_shares"][k]
            cpu_share = allocation["bs_cpu_shares"][k]
            if route is None or access_share <= 0 or cpu_share <= 0:
                task_times[k] = float("inf")
                usv_energies[k] = float("inf")
                continue

            access_rate = access_share * route["access_rate"]
            access_time = self.common.calculate_transmission_time(
                task_size, access_rate
            )
            total_time = access_time
            total_time += route["access_distance"] / self.base.light_speed
            usv_energies[k] = self.common.calculate_transmission_energy(
                usv["power"], access_time
            )

            feasible = np.isfinite(access_time)
            for hop_idx, hop in enumerate(route["backhaul_hops"]):
                share = allocation["backhaul_shares"].get((k, hop_idx), 0.0)
                if share <= 0:
                    feasible = False
                    break
                hop_time = self.common.calculate_transmission_time(
                    task_size, share * hop["rate"]
                )
                if not np.isfinite(hop_time):
                    feasible = False
                    break
                total_time += hop_time + hop["distance"] / self.base.light_speed
                uav_relay_energies[hop["tx_uav"]] += (
                    self.base.uav_relay_power * hop_time
                )

            if not feasible:
                task_times[k] = float("inf")
                usv_energies[k] = float("inf")
                continue

            bs_time = self.common.calculate_computation_time(
                task_size,
                task_resource,
                self.bs["resource"] * cpu_share,
            )
            task_times[k] = total_time + bs_time

        potential = float(np.sum(task_times))
        return {
            "task_times": task_times,
            "usv_energies": usv_energies,
            "uav_relay_energies": uav_relay_energies,
            "access_shares": allocation["access_shares"],
            "bs_cpu_shares": allocation["bs_cpu_shares"],
            "backhaul_shares": allocation["backhaul_shares"],
            "potential": potential,
        }

    def _run_potential_game(self, routes):
        """Find a binary profile that is locally minimal in total delay."""

        if self.base.potential_max_passes <= 0:
            raise RuntimeError("Potential game did not converge: invalid pass cap")
        decisions = np.asarray(self.offloading_decisions, dtype=int).copy()
        decisions[(np.array([route is None for route in routes]))] = 0
        tolerance = self.base.potential_tolerance
        self.last_potential_passes = 0
        self.last_potential_converged = False

        for pass_idx in range(self.base.potential_max_passes):
            changed = False
            for k in range(self.num_usvs):
                previous = int(decisions[k])
                local_profile = decisions.copy()
                local_profile[k] = 0
                local_value = self._evaluate_profile(
                    local_profile, routes
                )["potential"]

                if routes[k] is None:
                    new_decision = 0
                else:
                    offload_profile = decisions.copy()
                    offload_profile[k] = 1
                    offload_value = self._evaluate_profile(
                        offload_profile, routes
                    )["potential"]
                    if np.isfinite(offload_value) and (
                        offload_value < local_value - tolerance
                    ):
                        new_decision = 1
                    elif local_value < offload_value - tolerance:
                        new_decision = 0
                    else:
                        new_decision = previous

                decisions[k] = new_decision
                changed = changed or new_decision != previous

            self.last_potential_passes = pass_idx + 1
            if not changed:
                self.last_potential_converged = True
                break

        if not self.last_potential_converged:
            current_value = self._evaluate_profile(decisions, routes)["potential"]
            for k in range(self.num_usvs):
                alternative = decisions.copy()
                alternative[k] = 1 - alternative[k]
                if alternative[k] == 1 and routes[k] is None:
                    continue
                alternative_value = self._evaluate_profile(
                    alternative, routes
                )["potential"]
                if alternative_value < current_value - tolerance:
                    raise RuntimeError(
                        "Potential game did not converge within "
                        f"{self.base.potential_max_passes} passes"
                    )
            self.last_potential_converged = True

        return decisions.astype(int)

    def _count_nearby_usvs(self, uav_idx):
        count = 0
        uav_position = self.uavs[uav_idx]["position"]
        for usv in self.usvs:
            if np.linalg.norm(uav_position - usv["position"]) <= self.base.uav_coverage:
                count += 1
        return count

    def _calculate_stability_reward(self, uav_idx):
        if self.task_completion_rate >= 75:
            trajectory = self.uavs[uav_idx]["trajectory"]
            if len(trajectory) > 1:
                movement = np.linalg.norm(trajectory[-1] - trajectory[-2])
                covered_usvs = self._count_nearby_usvs(uav_idx)
                coverage_ratio = covered_usvs / self.num_usvs
                if coverage_ratio >= 0.15:
                    stability_base = 15.0
                    stability_multiplier = (
                        1.5 if self.task_completion_rate >= 80 else 1.0
                    )
                    stability_reward = (
                        stability_base
                        * stability_multiplier
                        / (1.0 + 2.0 * movement)
                    )
                    coverage_bonus = min(1.5, coverage_ratio * 3.0)
                    return stability_reward * coverage_bonus
        return 0.0

    def _calculate_coverage_reward(self, uav_idx):
        covered_usvs = self._count_nearby_usvs(uav_idx)
        coverage_ratio = covered_usvs / self.num_usvs
        if covered_usvs <= 0:
            return 0.0
        base_coverage_reward = 15.0 * np.sqrt(covered_usvs)
        movement_penalty = 0.0
        trajectory = self.uavs[uav_idx]["trajectory"]
        if len(trajectory) > 1 and coverage_ratio >= 0.3:
            movement = np.linalg.norm(trajectory[-1] - trajectory[-2])
            movement_penalty = min(0.2 * base_coverage_reward, 5.0 * movement)
        return max(0.0, base_coverage_reward - movement_penalty)

    def calculate_rewards(
        self,
        task_total_times,
        task_completion_rate,
        uav_service_energies,
        uav_fly_energies,
    ):
        """Copy the original reward arithmetic with relay service energy."""

        del task_total_times
        rewards = np.zeros(self.num_uavs)
        if task_completion_rate >= self.base.completion_threshold:
            normalized_rate = (
                task_completion_rate - self.base.completion_threshold
            ) / (100 - self.base.completion_threshold)
            completion_reward = 10.0 + 30.0 * normalized_rate
        else:
            completion_reward = 30.0 * (
                task_completion_rate / self.base.completion_threshold
            )

        for k in range(self.num_uavs):
            delay_penalty = -self.system_time
            energy_penalty = 0.0
            uav_total_energy = uav_service_energies[k] + uav_fly_energies[k]
            if uav_total_energy > self.base.energy_threshold:
                excess_ratio = min(
                    2.0, uav_total_energy / self.base.energy_threshold
                ) - 1.0
                energy_penalty = -20.0 * excess_ratio

            boundary_penalty = 0.0
            uav_position = self.uavs[k]["position"]
            if (
                uav_position[0] <= self.base.field_X[0]
                or uav_position[0] >= self.base.field_X[1]
                or uav_position[1] <= self.base.field_Y[0]
                or uav_position[1] >= self.base.field_Y[1]
            ):
                boundary_penalty = -100.0
            else:
                dist_x = min(
                    uav_position[0] - self.base.field_X[0],
                    self.base.field_X[1] - uav_position[0],
                )
                dist_y = min(
                    uav_position[1] - self.base.field_Y[0],
                    self.base.field_Y[1] - uav_position[1],
                )
                min_distance = min(dist_x, dist_y)
                if min_distance < self.base.boundary_margin:
                    boundary_penalty = -30.0 * (
                        1.0 - min_distance / self.base.boundary_margin
                    )

            collision_penalty = 0.0
            for other in range(self.num_uavs):
                if other == k:
                    continue
                distance = np.linalg.norm(
                    self.uavs[k]["position"] - self.uavs[other]["position"]
                )
                if distance < self.base.collision_threshold:
                    collision_penalty -= 20.0 * (
                        1.0 - distance / self.base.collision_threshold
                    )

            rewards[k] = (
                self.base.w_delay * delay_penalty
                + self.base.w_energy * energy_penalty
                + self.base.w_boundary * boundary_penalty
                + self.base.w_collision * collision_penalty
                + self.base.w_completion * completion_reward
                + self._calculate_stability_reward(k)
                + self._calculate_coverage_reward(k)
            )
        return rewards

    def _move_uavs(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_uavs, self.action_dim):
            raise ValueError(
                f"Expected action shape {(self.num_uavs, self.action_dim)}, "
                f"received {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("Actions must be finite")
        actions = np.clip(actions, -1.0, 1.0)
        fly_energies = np.zeros(self.num_uavs, dtype=float)
        for k, uav in enumerate(self.uavs):
            coverage_ratio = self._count_nearby_usvs(k) / self.num_usvs
            if self.task_completion_rate >= 90:
                max_speed, action_scale = 6.0, 0.3
            elif self.task_completion_rate >= 85:
                max_speed, action_scale = 9.0, 0.5
            elif self.task_completion_rate >= 70:
                max_speed, action_scale = 11.0, 0.6
            elif self.task_completion_rate >= 60:
                max_speed, action_scale = 13.0, 0.8
            else:
                max_speed, action_scale = 15.0, 1.0
            if coverage_ratio >= 0.3:
                max_speed = min(max_speed, 2.0)
                action_scale = 0.5

            velocity = actions[k] * action_scale * max_speed
            uav["position"] += velocity
            uav["position"][0] = np.clip(
                uav["position"][0], self.base.field_X[0], self.base.field_X[1]
            )
            uav["position"][1] = np.clip(
                uav["position"][1], self.base.field_Y[0], self.base.field_Y[1]
            )
            uav["trajectory"].append(uav["position"].copy())
            fly_energies[k] = self.common.get_uav_fly_energy(
                np.linalg.norm(velocity)
            )
        return fly_energies

    def step(self, actions):
        return self._run_with_rng(self._step, actions)

    def _step(self, actions):
        """Advance one multihop environment slot."""

        uav_fly_energies = self._move_uavs(actions)
        routes = self._build_routes()
        self.offloading_decisions = self._run_potential_game(routes)
        evaluation = self._evaluate_profile(self.offloading_decisions, routes)

        task_times = evaluation["task_times"]
        usv_energies = evaluation["usv_energies"]
        uav_relay_energies = evaluation["uav_relay_energies"]
        self.system_time = float(np.sum(task_times))
        total_usv_energy = float(np.sum(usv_energies))
        total_uav_energy = float(
            np.sum(uav_fly_energies) + np.sum(uav_relay_energies)
        )
        self.total_energy = total_usv_energy + total_uav_energy
        self.avg_usv_energy = total_usv_energy / self.num_usvs
        self.avg_uav_energy = total_uav_energy / self.num_uavs
        completed = int(np.sum(task_times <= self.base.task_completion_deadline))
        self.task_completion_rate = completed / self.num_usvs * 100.0

        reward_array = self.calculate_rewards(
            task_times,
            self.task_completion_rate,
            uav_relay_energies,
            uav_fly_energies,
        )
        rewards = reward_array.reshape(-1, 1)

        available_routes = sum(route is not None for route in routes)
        offloaded = np.where(self.offloading_decisions == 1)[0]
        hop_counts = [
            1 + len(routes[k]["backhaul_hops"])
            for k in offloaded
            if routes[k] is not None
        ]
        forwarded_tasks = np.zeros(self.num_uavs, dtype=int)
        for k in offloaded:
            if routes[k] is None:
                continue
            for tx_uav in {
                hop["tx_uav"] for hop in routes[k]["backhaul_hops"]
            }:
                forwarded_tasks[tx_uav] += 1

        self.common.get_usv_mobility(self.usvs)
        for usv in self.usvs:
            usv["task_size"] = np.random.randint(
                int(self.base.task_size_min), int(self.base.task_size_max)
            )
            usv["task_resource"] = np.random.randint(
                self.base.task_resources_min, self.base.task_resources_max
            )

        next_observation = self._get_observation()
        self.current_step += 1
        is_done = self.current_step >= self.base.episode_length
        dones = [is_done] * self.num_uavs
        info = {
            "system_time": self.system_time,
            "total_energy": self.total_energy,
            "avg_usv_energy": self.avg_usv_energy,
            "avg_uav_energy": self.avg_uav_energy,
            "completion_rate": self.task_completion_rate,
            "avg_uav_fly_energy": float(np.mean(uav_fly_energies)),
            "avg_uav_comp_energy": 0.0,
            "avg_uav_relay_energy": float(np.mean(uav_relay_energies)),
            "bs_offloading_ratio": len(offloaded) / self.num_usvs * 100.0,
            "route_availability_ratio": available_routes / self.num_usvs * 100.0,
            "avg_hop_count": float(np.mean(hop_counts)) if hop_counts else 0.0,
            "max_hop_count": int(max(hop_counts)) if hop_counts else 0,
            "uav_forwarded_tasks": forwarded_tasks.tolist(),
            "potential_passes": self.last_potential_passes,
            "potential_converged": self.last_potential_converged,
        }
        if is_done:
            info["trajectories"] = {
                "uavs": [uav["trajectory"] for uav in self.uavs],
                "usvs": [usv["trajectory"] for usv in self.usvs],
            }
        return next_observation, rewards, dones, info
