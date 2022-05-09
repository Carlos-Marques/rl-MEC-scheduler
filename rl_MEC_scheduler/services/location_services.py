import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy import spatial


def get_locations(object_with_positions):
    return np.array(
        tuple(
            object_with_position.location
            for object_with_position in object_with_positions
        )
    )


def loc_to_df(locations, type: str, n_dimensions: int) -> pd.DataFrame:
    if n_dimensions == 2:
        location_dict = {"x": locations[:, 0], "y": locations[:, 1]}
    else:
        location_dict = {
            "x": locations[:, 0],
            "y": locations[:, 1],
            "z": locations[:, 2],
        }
    df = pd.DataFrame(data=location_dict)
    df["type"] = type
    return df


def get_closest_action(
    UEs_locs: NDArray[np.float32], MECs_locs: NDArray[np.float32]
) -> NDArray[np.int64]:
    distance_matrix = spatial.distance.cdist(UEs_locs, MECs_locs)
    closest_MEC = np.argmin(distance_matrix, axis=1)
    action = closest_MEC + 1

    return action
