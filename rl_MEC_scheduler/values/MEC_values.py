from dataclasses import dataclass
from typing import Tuple


@dataclass
class MEC:
    frequency: float
    transmission_power: float
    location: Tuple[float, float, float]
