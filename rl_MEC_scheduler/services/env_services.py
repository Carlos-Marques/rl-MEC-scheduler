from typing import Tuple, Any
from collections import Counter
from itertools import chain

import numpy as np
from numpy.typing import NDArray

from rl_MEC_scheduler.values.MEC_values import MEC
from rl_MEC_scheduler.values.UE_values import UE
from rl_MEC_scheduler.values.network_values import Network
from rl_MEC_scheduler.values.task_values import TaskDistributions, Task
from rl_MEC_scheduler.services.simulation_services import (
    get_local_cost,
    get_offloading_cost,
)


def get_actions_cost(
    actions: NDArray[np.int64],
    tasks: Tuple[Task],
    UEs: Tuple[UE],
    MECs: Tuple[MEC],
    network: Network,
) -> float:
    n_tasks_per_MEC = Counter(actions)

    total_cost = 0
    for idx, action in enumerate(actions):
        if action == 0:
            total_cost += get_local_cost(task=tasks[idx], user_equipment=UEs[idx])
        else:
            total_cost += get_offloading_cost(
                task=tasks[idx],
                user_equipment=UEs[idx],
                mobile_edge_computer=MECs[action-1],
                n_tasks=n_tasks_per_MEC[action],
                network=network,
            )

    return total_cost


def generate_task(
    task_distributions: TaskDistributions, np_random: np.random.RandomState
) -> Task:
    input_data = np_random.normal(
        task_distributions.input_data_mean, task_distributions.input_data_std
    )
    output_data = np_random.normal(
        task_distributions.output_data_mean, task_distributions.output_data_std
    )
    n_cycles = np_random.normal(
        task_distributions.n_cycles_mean, task_distributions.n_cycles_std
    )

    return Task(
        input_data=input_data,
        output_data=output_data,
        n_cycles=n_cycles,
        energy_weight=task_distributions.energy_weight,
        delay_weight=task_distributions.delay_weight,
    )


def get_tasks(
    np_random: np.random.RandomState,
    task_distributions: TaskDistributions,
    n_tasks: int,
) -> Tuple[Task]:
    return tuple(
        generate_task(task_distributions=task_distributions, np_random=np_random)
        for _ in range(n_tasks)
    )


def get_observations(tasks: Tuple[Task], observation_size: int):
    flat_tasks = map(
        lambda task: [task.input_data, task.output_data, task.n_cycles, task.energy_weight, task.delay_weight], tasks
    )
    flat_tasks_chain = chain.from_iterable(flat_tasks)

    return np.fromiter(iter=flat_tasks_chain, dtype=np.float32, count=observation_size)


def get_UEs(
    np_random: np.random.RandomState,
    radius: float,
    n_UEs: int,
    frequency: float,
    tranmsission_power: float,
    idle_power: float,
    download_power: float,
    n_dimensions: int,
) -> Tuple[UE]:
    locations = np_random.random_sample((n_UEs, n_dimensions)) * radius

    return tuple(
        UE(
            frequency=frequency,
            transmission_power=tranmsission_power,
            idle_power=idle_power,
            download_power=download_power,
            location=tuple(location),
        )
        for location in locations
    )


def get_MECs(
    np_random: np.random.RandomState,
    radius: float,
    n_MECs: int,
    frequency: float,
    transmission_power: float,
    n_dimensions: int,
) -> Tuple[MEC]:
    locations = np_random.random_sample((n_MECs, n_dimensions)) * radius

    return tuple(
        MEC(
            frequency=frequency,
            transmission_power=transmission_power,
            location=tuple(location),
        )
        for location in locations
    )
