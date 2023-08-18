from typing import Tuple

import numpy as np


def filter_time_span(item, min_t, max_t):
    time_stamps = item["ball_time_stamps"]

    indices = []
    remove_indices = np.nonzero(min_t > time_stamps)[0]
    indices.extend(remove_indices)

    remove_indices = np.nonzero(time_stamps > max_t)[0]
    indices.extend(remove_indices)

    return indices


def filter_outside_region(
    item,
    xlimit: Tuple[float],
    ylimit: Tuple[float],
    zlimit: Tuple[float],
):
    limits = [xlimit, ylimit, zlimit]

    positions = item["ball_positions"]
    delete_indices = []

    for idx in range(len(positions)):
        for axis in range(3):
            if not (limits[axis][0] < positions[idx][axis] < limits[axis][1]):
                delete_indices.append(idx)
                break

    return delete_indices


def remove_samples(item, indices_to_remove):
    print(indices_to_remove)

    for key, values in item.items():
        if key == "launch_parameter":
            continue

        if key == "date_stamp":
            continue

        item = np.delete(values, indices_to_remove, axis=0)

    return item
