from dataclasses import dataclass

from rl_MEC_scheduler.services.network_services import get_band_capacity, get_error_coefficient

@dataclass
class Network:
    bandwidth: float
    n_subcarriers: int
    path_loss_exponent: float
    upload_channel_fading_coefficient: float
    download_channel_fading_coefficient: float
    upload_bit_error_rate: float
    download_bit_error_rate: float
    noise_power: float
    signal_noise_ratio: float
    n_devices: int

    def __post_init__(self):
        self.band_capacity = get_band_capacity(
            n_subcarriers=self.n_subcarriers,
            number_devices=self.n_devices,
            bandwidth=self.bandwidth,
        )
        self.upload_error_coefficient = get_error_coefficient(
            channel_fading_coefficient=self.upload_channel_fading_coefficient,
            noise_power=self.noise_power,
            signal_noise_ratio=self.signal_noise_ratio,
        )
        self.download_error_coefficient = get_error_coefficient(
            channel_fading_coefficient=self.download_channel_fading_coefficient,
            noise_power=self.noise_power,
            signal_noise_ratio=self.signal_noise_ratio,
        )