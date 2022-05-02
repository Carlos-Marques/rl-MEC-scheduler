def get_band_capacity(
    n_subcarriers: int, number_devices: int, bandwidth: float
) -> float:
    return n_subcarriers * (bandwidth / number_devices)


def get_error_coefficient(
    channel_fading_coefficient: float,
    noise_power: float,
    signal_noise_ratio: float,
) -> float:
    return (channel_fading_coefficient**2) / (signal_noise_ratio * noise_power)

