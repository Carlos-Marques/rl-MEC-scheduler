from typing import Tuple

import numpy as np
from numpy.linalg import norm

from rl_MEC_scheduler.values.UE_values import UE
from rl_MEC_scheduler.values.MEC_values import MEC
from rl_MEC_scheduler.values.network_values import Network
from rl_MEC_scheduler.values.task_values import Task


def get_execution_delay(n_cycles: float, frequency: float) -> float:
    return n_cycles / frequency


def get_execution_energy(n_cycles: float, consumption_per_cycle: float) -> float:
    return n_cycles * consumption_per_cycle


def get_cost(
    delay_weight: float,
    delay: float,
    energy_weight: float,
    energy: float,
) -> float:
    return delay_weight * delay + energy_weight * energy


def get_transmission_delay(n_bits: float, rate: float) -> float:
    return n_bits / rate


def get_distance(
    UE_location: Tuple[float, float, float], MEC_location: Tuple[float, float, float]
) -> float:
    return norm(np.array(UE_location) - np.array(MEC_location))  # type: ignore


def get_transmission_rate(
    band_capacity: float,
    error_coefficient: float,
    path_loss_exponent: float,
    transmission_power: float,
    distance: float,
) -> float:
    return band_capacity * np.log2(
        1 + (transmission_power * error_coefficient) / (distance**path_loss_exponent)
    )


def get_energy(delay: float, power: float) -> float:
    return delay * power


def get_frequency_per_task(
    frequency: float,
    n_tasks: int,
) -> float:
    return frequency / n_tasks


def get_total_offload_delay(
    execution_delay: float,
    transmission_delay: float,
    dowload_delay: float,
) -> float:
    return execution_delay + transmission_delay + dowload_delay


def get_total_offload_energy(
    idle_energy: float,
    transmission_energy: float,
    download_energy: float,
) -> float:
    return idle_energy + transmission_energy + download_energy


def get_local_cost(task: Task, user_equipment: UE) -> float:
    local_delay = get_execution_delay(
        n_cycles=task.n_cycles, frequency=user_equipment.frequency
    )
    local_energy = get_execution_energy(
        n_cycles=task.n_cycles,
        consumption_per_cycle=user_equipment.consumption_per_cycle,
    )

    return get_cost(
        delay_weight=task.delay_weight,
        delay=local_delay,
        energy_weight=task.energy_weight,
        energy=local_energy,
    )


def get_upload(
    task: Task, user_equipment: UE, network: Network, distance: float
) -> Tuple[float, float]:
    upload_rate = get_transmission_rate(
        band_capacity=network.band_capacity,
        error_coefficient=network.upload_error_coefficient,
        path_loss_exponent=network.path_loss_exponent,
        transmission_power=user_equipment.transmission_power,
        distance=distance,
    )

    upload_delay = get_transmission_delay(n_bits=task.input_data, rate=upload_rate)
    upload_energy = get_energy(
        delay=upload_delay, power=user_equipment.transmission_power
    )

    return upload_delay, upload_energy


def get_download(
    task: Task,
    user_equipment: UE,
    mobile_edge_computer: MEC,
    network: Network,
    distance: float,
) -> Tuple[float, float]:
    dowload_rate = get_transmission_rate(
        band_capacity=network.band_capacity,
        error_coefficient=network.download_error_coefficient,
        path_loss_exponent=network.path_loss_exponent,
        transmission_power=mobile_edge_computer.transmission_power,
        distance=distance,
    )
    download_delay = get_transmission_delay(n_bits=task.output_data, rate=dowload_rate)
    download_energy = get_energy(
        delay=download_delay, power=user_equipment.download_power
    )

    return download_delay, download_energy


def get_offloading_cost(
    task: Task,
    user_equipment: UE,
    mobile_edge_computer: MEC,
    n_tasks: int,
    network: Network,
) -> float:
    distance = get_distance(
        UE_location=user_equipment.location, MEC_location=mobile_edge_computer.location
    )

    upload_delay, upload_energy = get_upload(
        task=task, user_equipment=user_equipment, network=network, distance=distance
    )

    frequency_per_task = get_frequency_per_task(
        frequency=mobile_edge_computer.frequency, n_tasks=n_tasks
    )
    execution_delay = get_execution_delay(
        n_cycles=task.n_cycles, frequency=frequency_per_task
    )
    idle_energy = get_energy(delay=execution_delay, power=user_equipment.idle_power)

    download_delay, download_energy = get_download(
        task=task,
        user_equipment=user_equipment,
        mobile_edge_computer=mobile_edge_computer,
        network=network,
        distance=distance,
    )

    total_delay = upload_delay + execution_delay + download_delay
    total_energy = upload_energy + idle_energy + download_energy

    total_offloading_cost = get_cost(
        delay_weight=task.delay_weight,
        delay=total_delay,
        energy_weight=task.energy_weight,
        energy=total_energy,
    )

    return total_offloading_cost
