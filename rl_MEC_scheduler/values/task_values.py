from dataclasses import dataclass


@dataclass
class TaskDistributions:
    input_data_mean: float
    input_data_std: float
    output_data_mean: float
    output_data_std: float
    n_cycles_mean: float
    n_cycles_std: float
    energy_weight: float
    delay_weight: float


@dataclass
class Task:
    input_data: float
    output_data: float
    n_cycles: float
    energy_weight: float
    delay_weight: float
