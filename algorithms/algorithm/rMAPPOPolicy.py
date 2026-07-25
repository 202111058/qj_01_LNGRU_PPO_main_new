"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : rMAPPOPolicy.py
"""

import torch
import numpy as np
from algorithms.algorithm.r_actor_critic import R_Actor, R_Critic
from utils.util import update_linear_schedule


class RMAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def _correct_rnn_states(self, obs, rnn_states):
        """
        Helper function to correct rnn_states batch size to match obs batch size.
        """
        if isinstance(rnn_states, np.ndarray):
            rnn_states = torch.from_numpy(rnn_states).to(self.device).float()

        # [关键修复] 批次大小在第 0 维
        obs_batch_size = obs.shape[0]
        rnn_batch_size = rnn_states.shape[0]

        if obs_batch_size != rnn_batch_size:
            if obs_batch_size > rnn_batch_size:
                # 放大: 从 (1, L, H) -> (N, L, H)
                # 我们取第一个 hidden state 然后重复 N 次
                rnn_states = rnn_states[0:1].repeat(obs_batch_size, 1, 1)
            else:
                # 缩小: 从 (N, L, H) -> (M, L, H)
                rnn_states = rnn_states[:obs_batch_size, :, :]

        return rnn_states

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        """
        rnn_states_actor = self._correct_rnn_states(obs, rnn_states_actor)
        rnn_states_critic = self._correct_rnn_states(cent_obs, rnn_states_critic)

        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        """
        rnn_states_critic = self._correct_rnn_states(cent_obs, rnn_states_critic)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        """
        rnn_states_actor = self._correct_rnn_states(obs, rnn_states_actor)
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor