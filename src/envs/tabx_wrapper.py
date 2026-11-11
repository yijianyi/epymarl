"""EPyMARL's fixed-shape TABX environment with explicit episode boundaries."""
import numpy as np
from functools import lru_cache

from .multiagentenv import MultiAgentEnv
from .tabx_layout import TABXLayout
from .tabx_outcome import adjudicate, replace_terminal_reward
from .tabx_tasks import build_task_params, load_task, training_episode_seed


@lru_cache(maxsize=32)
def _simulator_functions(episode_limit, jit, n_ally, n_enemy, n_zone):
    import jax
    from src.tabx import TABX
    from src.tabx.config import TABXConfig
    from src.tabx.wrappers.wrappers import TABXEnemyHeuristicWrapper
    base = TABX(TABXConfig(max_n_ally=n_ally, max_n_enemy=n_enemy, max_n_zone=n_zone), world_state_type="global",
                max_episode_steps=episode_limit, position_permutation=False)
    simulator = TABXEnemyHeuristicWrapper(base)
    return (base, simulator, jax.jit(simulator.reset) if jit else simulator.reset,
            jax.jit(simulator.step) if jit else simulator.step)


class TABXWrapper(MultiAgentEnv):
    n_agents = 10
    IDLE = 7

    def __init__(self, key, task_bank=None, timeout_rule="hp", max_episode_steps=512,
                 world_state_type="global", seed=0, common_reward=True,
                 reward_scalarisation="sum", tabx_root=None, expected_task_hash=None,
                 expected_bank_hash=None, jit=True, opponent_difficulty="medium"):
        if not common_reward or reward_scalarisation != "sum":
            raise ValueError("TABX uses one common team reward with scalarisation=sum")
        if timeout_rule not in ("hp", "loss"):
            raise ValueError("timeout_rule must be hp or loss")
        if world_state_type != "global" or not 1 <= int(max_episode_steps) <= 512:
            raise ValueError("TABX requires global state and a limit in [1,512]")
        self.task, self.manifest = load_task(task_bank, key, expected_task_hash, expected_bank_hash)
        self.opponent_difficulty = opponent_difficulty
        self.params, cfg = build_task_params(self.task, tabx_root, opponent_difficulty)
        self.layout = TABXLayout(cfg.max_n_ally, cfg.max_n_enemy, cfg.max_n_zone)
        self.manifest.update(self.layout.metadata())
        import jax

        self.jax = jax
        self.episode_limit = int(max_episode_steps)
        self.rule = timeout_rule
        self.base_env, self.simulator, self._reset, self._step = _simulator_functions(
            self.episode_limit, bool(jit), self.layout.n_ally, self.layout.n_enemy, self.layout.n_zone)
        self.seed(seed)
        self.env_state = None
        self.ended = True
        assert self.base_env.world_state_size() == self.layout.native_state_size
        assert self.base_env.observation_spaces[self.base_env.ally_keys[0]].shape == (self.layout.native_obs_size,)

    def seed(self, seed=None):
        self._train_seed = 0 if seed is None else int(seed)
        self._episode_index = 0

    def _next_key(self):
        self._episode_key, key = self.jax.random.split(self._episode_key)
        return key

    def reset(self, seed=None, options=None):
        if seed is None:
            self.last_train_episode_index = self._episode_index
            seed = training_episode_seed(self.manifest["task_hash"], self._train_seed, self._episode_index)
            self._episode_index += 1
        else:
            self.last_train_episode_index = None
        self.last_episode_seed = int(seed)
        self._episode_key = self.jax.random.PRNGKey(self.last_episode_seed)
        obs, self.env_state = self._reset(self._next_key(), self.params)
        self.t = 0
        self.ended = False
        self.first_kill = 0  # 0 none; 1 ally; 2 enemy; 3 simultaneous
        self.last_info = {}
        self._cache(obs)
        self._previous_alive = self._native_alive.copy()
        return self.get_obs(), {}

    def _cache(self, obs):
        self._native_obs = {key: np.asarray(value, dtype=np.float32).copy() for key, value in obs.items()}
        self._obs = self.layout.pack_observations([obs[k] for k in self.base_env.ally_keys])
        self._state = self.layout.pack_state(obs["world_state"])
        state = self.env_state["state"]
        self._present_all = self.layout.present
        self._native_alive = np.array([bool(np.asarray(state[k].status.is_alive).item())
                                    and not bool(np.asarray(state[k].status.is_disabled).item())
                                    and float(np.asarray(state[k].status.health).item()) > 0
                                    for k in self.base_env.unit_keys])
        self._all_alive = self.layout.pack_alive(self._native_alive).astype(bool)
        avail = self.simulator.get_avail_actions(self.env_state)
        self._native_avail = np.asarray([avail[k] for k in self.base_env.ally_keys], dtype=np.int32)
        self._avail = self.layout.pack_avail_actions(self._native_avail, self._native_alive[:self.layout.n_ally])
        if (self._obs.shape != (10, 378) or self._state.shape != (340,) or
                self._avail.shape != (10, 8) or not np.all(self._avail.sum(axis=1) > 0)):
            raise ValueError("Unexpected TABX observation/state/action shape")
        if not np.isfinite(self._obs).all() or not np.isfinite(self._state).all():
            raise ValueError("Non-finite TABX observation/state")

    def step(self, actions):
        if self.ended:
            raise RuntimeError("reset() is required before stepping a completed episode")
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        array = np.asarray(actions).reshape(-1)
        if array.size != 10 or not np.all(np.isfinite(array)) or not np.all(array == array.astype(np.int64)):
            raise ValueError(f"Expected exactly 10 integer actions, got {actions}")
        array = array.astype(np.int64)
        if np.any((array < 0) | (array >= 8)) or not self._avail[np.arange(10), array].all():
            raise ValueError(f"Illegal TABX action task={self.manifest['task_id']} t={self.t}: {array.tolist()}")
        native_actions = self.layout.native_actions(array)
        action_dict = {key: self.jax.numpy.asarray(native_actions[i]) for i, key in enumerate(self.base_env.ally_keys)}
        obs, self.env_state, rewards, _, raw_info = self._step(self._next_key(), self.env_state, action_dict)
        self._cache(obs)
        manager = self.env_state["state"]["game_manager"]
        self.t = int(np.asarray(manager.timestep).item())
        hp = np.asarray(manager.team_hp_ratio).reshape(-1)
        if hp.shape != (2,) or not np.isfinite(hp).all():
            raise ValueError("Invalid TABX team HP ratios")
        outcome = adjudicate(self._native_alive[:self.layout.n_ally].any(), self._native_alive[self.layout.n_ally:].any(),
                             float(hp[0]), float(hp[1]), self.t, self.episode_limit, self.rule,
                             self.base_env.win_reward, self.base_env.lose_reward)
        reward, dense = replace_terminal_reward(
            np.asarray(rewards["__all__"]).item(), np.asarray(raw_info["done_reward"]).reshape(-1)[0], outcome)
        if not np.isfinite(reward):
            raise ValueError("Non-finite TABX reward")
        died = self._previous_alive & ~self._native_alive
        if not self.first_kill and died.any():
            self.first_kill = int(died[self.layout.n_ally:].any()) + 2 * int(died[:self.layout.n_ally].any())
        self._previous_alive = self._native_alive.copy()
        self.ended = outcome.ended
        self.last_info = dict(episode_limit=False, hit_time_limit=int(outcome.hit_time_limit),
                              is_win=int(outcome.is_win), battle_won=int(outcome.is_win),
                              end_reason=outcome.reason, ally_hp_ratio=float(hp[0]), enemy_hp_ratio=float(hp[1]),
                              elimination_win=int(outcome.is_win and outcome.reason == 1),
                              timeout_win=int(outcome.is_win and outcome.reason == 2),
                              pure_timeout=int(outcome.reason == 2),
                              first_kill_ally=int(self.first_kill == 1),
                              first_kill_enemy=int(self.first_kill == 2),
                              first_kill_simultaneous=int(self.first_kill == 3))
        self.last_dense_reward = dense
        return self.get_obs(), float(reward), outcome.ended, False, dict(self.last_info)

    def get_obs(self):
        return self._obs.copy()

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id].copy()

    def get_state(self):
        return self._state.copy()

    def get_avail_actions(self):
        return self._avail.copy()

    def get_avail_agent_actions(self, agent_id):
        return self._avail[agent_id].copy()

    def get_obs_size(self):
        return 378

    def get_state_size(self):
        return 340

    def get_total_actions(self):
        return 8

    def get_frame(self):
        return dict(obs=self.get_obs(), state=self.get_state(), avail_actions=self.get_avail_actions(),
                    present=self._present_all[:10].astype(np.uint8), alive=self._all_alive[:10].astype(np.uint8))

    def get_native_observations(self):
        return {key: value.copy() for key, value in self._native_obs.items()}

    def get_layout_info(self):
        return self.layout.metadata()

    def close(self):
        pass

    def save_replay(self):
        raise NotImplementedError("Use tabx_pipeline collect to save trajectories")

    def render(self):
        raise NotImplementedError("TABX pipeline supports headless rollout")
