import os
import json
from dataclasses import asdict
from typing import Tuple, Any, Type

def save_configs(
    configs: Tuple[Any],
    configs_path: str,
    configs_filename: str,
) -> None:
    configs_dict = tuple(asdict(config) for config in configs)

    with open(os.path.join(configs_path, configs_filename), "w") as f:
        json.dump(configs_dict, f, indent=4, sort_keys=True)


def load_configs(
    config_type: Type, configs_path: str, configs_filename: str
) -> Tuple[Type]:
    with open(os.path.join(configs_path, configs_filename), "r") as f:
        configs_dict = json.load(f)

    configs = tuple(config_type(**config) for config in configs_dict)

    return configs

