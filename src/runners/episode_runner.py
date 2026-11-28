from functools import partial

import numpy as np
import torch as th

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.record_agent_alive = "agent_alive" in scheme
        self.record_ppo_behavior = "behavior_log_prob" in scheme
        self.teacher = None
        if self.record_ppo_behavior and self.args.tabx_mix_start:
            from tabx_expert import HeuristicTeacher
            self.teacher = HeuristicTeacher(self.env)

    def start_ppo_batch(self):
        from tabx_config import rollout_schedule
        self.batch_start_t_env = self.t_env
        self.teacher_probability, self.policy_temperature = rollout_schedule(self.args, self.batch_start_t_env)

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, episode_seed=None):
        self.batch = self.new_batch()
        if episode_seed is None:
            self.env.reset()
        else:
            self.env.reset(seed=episode_seed)
        self.t = 0

    def _state_data(self):
        data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if self.record_agent_alive:
            data["agent_alive"] = [self.env.get_agent_alive()]
        return data

    @th.no_grad()
    def run(self, test_mode=False, *, episode_seed=None, recorder=None):
        self.reset(episode_seed=episode_seed)
        if recorder is not None:
            recorder.start(self.env.get_frame(), episode_seed)

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)
        record_behavior = self.record_ppo_behavior and not test_mode
        teacher_steps = 0
        if record_behavior:
            block_size = self.args.tabx_mix_block_steps
            if self.teacher_probability:
                seed = self.env.last_episode_seed
                self.teacher.reset(seed)
                control_rng = np.random.default_rng([seed, 0x4D4958])

        while not terminated:
            self.batch.update(self._state_data(), ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            behavior = {}
            if record_behavior:
                offset = self.t % block_size
                if offset == 0:
                    teacher_control = bool(self.teacher_probability and control_rng.random() < self.teacher_probability)
                log_prob = self.mac.policy_log_probs(self.batch, self.t, self.policy_temperature)
                teacher_actions = self.teacher.actions() if self.teacher_probability else None
                if teacher_control:
                    actions = th.as_tensor(teacher_actions, device=self.args.device).unsqueeze(0)
                    taken_log_prob = th.zeros_like(actions, dtype=th.float).unsqueeze(-1)
                    teacher_steps += 1
                else:
                    actions = self.mac.action_selector.select_action(
                        log_prob.exp(), self.batch["avail_actions"][:, self.t], self.t_env
                    )
                    taken_log_prob = log_prob.gather(-1, actions.unsqueeze(-1))
                behavior = dict(teacher_control=[[teacher_control]], control_remaining=[[block_size - offset]],
                                policy_temperature=[[self.policy_temperature]], behavior_log_prob=taken_log_prob)
            else:
                actions = self.mac.select_actions(
                    self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
                )

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                **behavior,
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)
            if recorder is not None:
                recorder.step(actions[0], reward, post_transition_data["terminated"][0][0],
                              self.env.get_frame(), env_info)

            self.t += 1

        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(self._state_data(), ts=self.t)

        # Preserve the final-state action in the batch without drawing a PPO training sample.
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode or self.record_agent_alive
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        self.last_episode_info = dict(env_info, length=self.t, return_value=float(np.sum(episode_return)))
        if record_behavior:
            self.last_episode_info.update(teacher_steps=teacher_steps, teacher_fraction=teacher_steps / self.t,
                                          teacher_probability=self.teacher_probability,
                                          policy_temperature=self.policy_temperature)
        if recorder is not None:
            recorder.finish(self.last_episode_info)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if not self.record_agent_alive and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        if self.args.common_reward:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_mean",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_return_std", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
