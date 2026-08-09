"""Continuous-action wrapper for the standalone multihop UAV-BS core."""

import gym
import numpy as np
from gym import spaces

from envs.envs_multihop_bs import EnvCore


class ContinuousMultihopEnv:
    """Expose the multihop core through the existing MAPPO environment API."""

    def __init__(self, env_overrides=None):
        self.env = EnvCore(overrides=env_overrides)
        self.num_agent = self.env.num_uavs
        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim
        self.discrete_action_input = False
        self.movable = True

        self.action_space = [
            spaces.Box(
                low=-1,
                high=1,
                shape=(self.signal_action_dim,),
                dtype=np.float32,
            )
            for _ in range(self.num_agent)
        ]
        self.observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.signal_obs_dim,),
                dtype=np.float32,
            )
            for _ in range(self.num_agent)
        ]
        shared_dimension = self.signal_obs_dim * self.num_agent
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(shared_dimension,),
                dtype=np.float32,
            )
            for _ in range(self.num_agent)
        ]

    def _validate_transition(self, observations, rewards, dones):
        observations = np.asarray(observations, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=float)
        dones = np.asarray(dones, dtype=bool)
        expected_obs = (self.num_agent, self.signal_obs_dim)
        expected_rewards = (self.num_agent, 1)
        expected_dones = (self.num_agent,)
        if observations.shape != expected_obs:
            raise ValueError(
                f"Expected observation shape {expected_obs}, "
                f"received {observations.shape}"
            )
        if rewards.shape != expected_rewards:
            raise ValueError(
                f"Expected reward shape {expected_rewards}, received {rewards.shape}"
            )
        if dones.shape != expected_dones:
            raise ValueError(
                f"Expected done shape {expected_dones}, received {dones.shape}"
            )
        if not np.all(np.isfinite(observations)):
            raise ValueError("Observations must be finite")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("Rewards must be finite")
        return observations, rewards, dones

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        expected_actions = (self.num_agent, self.signal_action_dim)
        if actions.shape != expected_actions:
            raise ValueError(
                f"Expected action shape {expected_actions}, received {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("Actions must be finite")
        # The policy distribution is Gaussian and can occasionally sample
        # outside the declared Box.  Enforce the physical action contract.
        actions = np.clip(actions, -1.0, 1.0)
        observations, rewards, dones, info = self.env.step(actions)
        observations, rewards, dones = self._validate_transition(
            observations, rewards, dones
        )
        return observations, rewards, dones, info

    def reset(self):
        observations = np.asarray(self.env.reset(), dtype=np.float32)
        expected = (self.num_agent, self.signal_obs_dim)
        if observations.shape != expected:
            raise ValueError(
                f"Expected reset observation shape {expected}, "
                f"received {observations.shape}"
            )
        if not np.all(np.isfinite(observations)):
            raise ValueError("Reset observations must be finite")
        return observations

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        del mode
        return None

    def seed(self, seed):
        return self.env.seed(seed)
