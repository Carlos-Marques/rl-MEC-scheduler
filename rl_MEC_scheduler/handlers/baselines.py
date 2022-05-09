from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from rl_MEC_scheduler.handlers.env_handler import NetworkEnv
from rl_MEC_scheduler.services.location_services import (
    get_closest_action,
    get_locations,
)


class ClosestActionBaseline:
    def __init__(
        self, UEs_locs: NDArray[np.float32], MECs_locs: NDArray[np.float32]
    ):
        self._action = get_closest_action(
            UEs_locs=UEs_locs, MECs_locs=MECs_locs
        )

    def get_action(self):
        return self._action


class LocalActionBaseline:
    def __init__(self, shape: tuple):
        self._action = np.zeros(shape=shape, dtype=np.int64)

    def get_action(self):
        return self._action


def calculate_baseline(
    env: NetworkEnv,
    n_episodes: int,
    n_steps: int,
    get_action: Callable[[], NDArray[np.int64]],
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Calculate the baseline for the given environment.
    """
    rewards = np.zeros([n_episodes])

    for episode_n in range(n_episodes):
        observation = env.reset()
        for _ in range(n_steps):
            action = get_action()
            observation, reward, done, info = env.step(action)
            rewards[episode_n] += reward

    return rewards.min(), rewards.mean(), rewards.max()


def get_baselines(
    env: NetworkEnv, n_episodes: int, n_steps: int
) -> Tuple[
    Tuple[NDArray, NDArray, NDArray],
    Tuple[NDArray, NDArray, NDArray],
    Tuple[NDArray, NDArray, NDArray],
]:
    UEs_locs = get_locations(env.UEs)
    MECs_locs = get_locations(env.MECs)

    assert env.action_space.shape  # type: ignore
    local_rewards = calculate_baseline(
        env=env,
        n_episodes=n_episodes,
        n_steps=n_steps,
        get_action=LocalActionBaseline(shape=env.action_space.shape).get_action,  # type: ignore
    )

    closest_rewards = calculate_baseline(
        env=env,
        n_episodes=n_episodes,
        n_steps=n_steps,
        get_action=ClosestActionBaseline(UEs_locs, MECs_locs).get_action,
    )

    random_rewards = calculate_baseline(
        env=env,
        n_episodes=n_episodes,
        n_steps=n_steps,
        get_action=env.action_space.sample,
    )

    return local_rewards, closest_rewards, random_rewards
