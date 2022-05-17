from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

import numpy as np
from numpy.typing import NDArray
import gym
from gym import spaces
from gym.utils import seeding

from rl_MEC_scheduler.services.env_services import (
    get_actions_cost,
    get_observations,
    get_reward,
    get_tasks,
)
from rl_MEC_scheduler.values.MEC_values import MEC
from rl_MEC_scheduler.values.UE_values import UE
from rl_MEC_scheduler.values.network_values import Network
from rl_MEC_scheduler.values.task_values import TaskDistributions
from rl_MEC_scheduler.repository.config_repository import load_configs


@dataclass
class NetworkEnv(gym.Env):
    seed_value: Optional[int]
    UEs: Tuple[UE]
    MECs: Tuple[MEC]
    network: Network
    task_distributions: TaskDistributions
    mean_weight: float
    max_weight: float

    def __post_init__(self):
        self.seed(seed=self.seed_value)

        self.n_UEs = len(self.UEs)
        self.n_MECs = len(self.MECs)
        self.n_observations = len(self.UEs) * 5
        n_actions = self.n_MECs + 1

        self.action_space = spaces.MultiDiscrete([n_actions] * self.n_UEs, seed=self.seed_value) # type: ignore
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_observations,),
            dtype=np.float32,
            seed=self.seed_value, # type: ignore
        )

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> NDArray[np.float32]:
        self.tasks = get_tasks(
            np_random=self.np_random,
            task_distributions=self.task_distributions,
            n_tasks=self.n_UEs,
        )

        return get_observations(
            tasks=self.tasks, observation_size=self.n_observations
        )

    def step(
        self, action: NDArray[np.int64]
    ) -> Tuple[NDArray[np.float32], float, bool, dict]:
        assert self.action_space.contains(action)

        actions_cost = get_actions_cost(
            actions=action,
            tasks=self.tasks,
            UEs=self.UEs,
            MECs=self.MECs,
            network=self.network,
        )

        reward = get_reward(
            actions_cost=actions_cost,
            n_UEs=self.n_UEs,
            mean_weight=self.mean_weight,
            max_weight=self.max_weight,
        )

        self.tasks = get_tasks(
            np_random=self.np_random,
            task_distributions=self.task_distributions,
            n_tasks=self.n_UEs,
        )
        observation = get_observations(
            tasks=self.tasks, observation_size=self.n_observations
        )
        assert self.observation_space.contains(observation)

        done = False
        info = {"actions_cost": actions_cost}

        return observation, reward, done, info


def locations_to_tuple(objects: Tuple[Any]) -> Tuple[Any]:
    for object in objects:
        object.location = tuple(object.location)
    return objects


def format_env(env: Any) -> NetworkEnv:
    env.network = Network(**env.network)
    env.UEs = tuple(UE(**ue) for ue in env.UEs)
    locations_to_tuple(env.UEs)
    env.MECs = tuple(MEC(**mec) for mec in env.MECs)
    locations_to_tuple(env.MECs)
    env.task_distributions = TaskDistributions(**env.task_distributions)

    return env


def load_envs(config_path: str, config_filename: str) -> Tuple[NetworkEnv]:
    loaded_envs = load_configs(
        config_type=NetworkEnv,
        configs_path=config_path,
        configs_filename=config_filename,
    )
    return tuple(format_env(env) for env in loaded_envs)
