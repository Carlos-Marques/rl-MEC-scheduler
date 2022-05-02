
from dataclasses import dataclass
from typing import Tuple

@dataclass
class UE:
    frequency: float
    transmission_power: float
    idle_power: float
    download_power: float
    location: Tuple[float, float, float]

    def __post_init__(self):
        self.consumption_per_cycle = 10e-27 * (self.frequency**2)
