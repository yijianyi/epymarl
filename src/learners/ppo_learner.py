# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy
from pathlib import Path

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry


class PPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.tabx_rollouts = "agent_alive" in scheme
        self.use_gae = self.tabx_rollouts and getattr(args, "use_gae", False)

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            value_shape = (1,) if self.tabx_rollouts else (self.n_agents,)
            self.ret_ms = RunningMeanStd(shape=value_shape, device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        if self.tabx_rollouts:
            # The terminal observation is filled but has no successor.
            filled = batch["filled"].bool()
            transition_mask = filled[:, :-1] & filled[:, 1:]
            alive_at_action = batch["agent_alive"][:, :-1].squeeze(-1)
            actor_mask = (transition_mask & alive_at_action).float()
            critic_mask = transition_mask
        else:
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            actor_mask = mask.repeat(1, 1, self.n_agents)
            critic_mask = actor_mask

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            if not self.tabx_rollouts:
                rewards = rewards.expand(-1, -1, self.n_agents)

        actor_count = actor_mask.sum()
        gae_advantages = None
        gae_targets = None
        with th.no_grad():
            if self.use_gae:
                alive = batch["agent_alive"].squeeze(-1)
                team_values = self._tabx_team_values(self.critic(batch).squeeze(-1), alive)
                if self.args.standardise_returns:
                    team_values = team_values * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
                gae_advantages, gae_targets = self._tabx_gae(
                    rewards, transition_mask, batch["terminated"][:, :-1].bool(), team_values
                )
                if self.args.standardise_returns:
                    gae_targets = self._standardise_targets(gae_targets)

            old_mac_out = []
            self.old_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.old_mac.forward(batch, t=t)
                old_mac_out.append(agent_outs)
            old_pi = th.stack(old_mac_out, dim=1)
            old_pi[actor_mask == 0] = 1.0
            old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
            old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        for _ in range(self.args.epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            pi = mac_out
            critic_advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask,
                target_returns=gae_targets,
            )
            advantages = gae_advantages if self.use_gae else critic_advantages
            advantages = advantages.detach().expand_as(actor_mask)
            # Calculate policy grad with mask

            pi[actor_mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            log_ratios = log_pi_taken - old_log_pi_taken
            ratios = log_ratios.exp()
            surr1 = ratios * advantages
            surr2 = (
                th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * advantages
            )

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = (
                -(
                    (th.min(surr1, surr2) + self.args.entropy_coef * entropy) * actor_mask
                ).sum()
                / actor_count
            )

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.training_steps += 1
        if not self.use_gae:
            if (
                self.args.target_update_interval_or_tau > 1
                and (self.training_steps - self.last_target_update_step)
                / self.args.target_update_interval_or_tau
                >= 1.0
            ):
                self._update_targets_hard()
                self.last_target_update_step = self.training_steps
            elif self.args.target_update_interval_or_tau <= 1.0:
                self._update_targets_soft(self.args.target_update_interval_or_tau)

        ppo_stats = {}
        if self.tabx_rollouts:
            # Diagnostics use the policy and critic losses from the final PPO epoch.
            with th.no_grad():
                approx_kl = log_ratios.expm1() - log_ratios
                clipped = (ratios - 1).abs() > self.args.eps_clip
                ppo_stats = {
                    "entropy": ((entropy * actor_mask).sum() / actor_count).item(),
                    "approx_kl": ((approx_kl * actor_mask).sum() / actor_count).item(),
                    "clip_fraction": ((clipped * actor_mask).sum() / actor_count).item(),
                }
            if getattr(self.args, "tabx_run_dir", None):
                from tabx_reporting import PPO_TRAIN_FIELDS, append_train_metrics
                append_train_metrics(
                    Path(self.args.tabx_run_dir) / "train_metrics.csv",
                    dict(t_env=t_env, learner_updates=self.training_steps,
                         rollout_episodes=batch.batch_size, policy_loss=pg_loss.item(),
                         critic_loss=critic_train_stats["critic_loss"][0], **ppo_stats),
                    fields=PPO_TRAIN_FIELDS,
                )

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "advantage_mean",
                ((advantages * actor_mask).sum() / actor_count).item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                ((pi.max(dim=-1)[0] * actor_mask).sum() / actor_count).item(),
                t_env,
            )
            for key, value in ppo_stats.items():
                self.logger.log_stat(key, value, t_env)
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask, *,
                                target_returns=None):
        alive = batch["agent_alive"].squeeze(-1) if self.tabx_rollouts else None
        if target_returns is None:
            with th.no_grad():
                target_vals = target_critic(batch).squeeze(3)
                if alive is not None:
                    target_vals = self._tabx_team_values(target_vals, alive)

            if self.args.standardise_returns:
                target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

            if self.tabx_rollouts:
                target_returns = self.tabx_nstep_returns(
                    rewards, mask, batch["terminated"][:, :-1].bool(), target_vals
                )
            else:
                target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
            if self.args.standardise_returns:
                target_returns = self._standardise_targets(target_returns)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        if alive is not None:
            v = self._tabx_team_values(v, alive[:, :-1])
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )

        return masked_td_error, running_log

    def _standardise_targets(self, targets):
        self.ret_ms.update(targets)
        return (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

    @staticmethod
    def _tabx_team_values(values, alive):
        live_count = alive.sum(dim=-1, keepdim=True).clamp_min(1)
        return values.masked_fill(~alive, 0).sum(dim=-1, keepdim=True) / live_count

    def _tabx_gae(self, rewards, transition_mask, terminated, team_values):
        continuation = transition_mask & ~terminated
        state_values = team_values[:, :-1]
        next_values = team_values[:, 1:].masked_fill(~continuation, 0)
        deltas = rewards + self.args.gamma * next_values - state_values
        deltas = deltas.masked_fill(~transition_mask, 0)

        trace_decay = (self.args.gamma * self.args.gae_lambda) * continuation
        advantages = th.zeros_like(deltas)
        next_advantage = th.zeros_like(team_values[:, 0])
        for t in reversed(range(rewards.size(1))):
            next_advantage = deltas[:, t] + trace_decay[:, t] * next_advantage
            advantages[:, t] = next_advantage

        targets = (state_values + advantages).masked_fill(~transition_mask, 0)
        return advantages, targets

    def tabx_nstep_returns(self, rewards, transition_mask, terminated, team_values):
        # Team value follows the episode through individual casualties.
        continuation = transition_mask & ~terminated
        returns = team_values
        padding = th.zeros_like(returns[:, :1])
        for _ in range(self.args.q_nstep):
            targets = (rewards + self.args.gamma * returns[:, 1:] * continuation) * transition_mask
            returns = th.cat((targets, padding), dim=1)
        return returns[:, :-1]

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += (
                        self.args.gamma ** (step) * values[:, t] * mask[:, t]
                    )
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += (
                        self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += (
                        self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.old_mac.load_state(self.mac)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
