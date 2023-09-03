from typing import Any, Dict, List, Tuple

import numpy as np


def filter_time_span(data: dict, min_t: float, max_t: float) -> List[Dict[str, Any]]:
    new_data = data.copy()

    general_keys = ["launch_parameter", "date_stamp", "time_stamps"]

    ball_keys = [
        "ball_ids",
        "ball_time_stamps",
        "ball_positions",
        "ball_velocities",
    ]

    robot_keys = [
        "robot_time_stamps",
        "robot_joint_angles",
        "robot_joint_angles_desired",
        "robot_joint_angle_velocities",
        "robot_joint_angle_velocities_desired",
        "robot_pressures_agonist",
        "robot_pressures_antagonist",
    ]

    # filter ball data
    ball_time_stamps = data["ball_time_stamps"]

    indices = []

    remove_indices = np.nonzero(min_t > ball_time_stamps)[0]
    indices.extend(remove_indices)

    remove_indices = np.nonzero(ball_time_stamps > max_t)[0]
    indices.extend(remove_indices)

    for key in ball_keys:
        val = new_data[key]
        filtered_val = np.delete(val, indices, axis=0)
        new_data[key] = filtered_val

    # filter robot data
    robot_time_stamps = data["robot_time_stamps"]

    indices = []

    remove_indices = np.nonzero(min_t > robot_time_stamps)[0]
    indices.extend(remove_indices)

    remove_indices = np.nonzero(robot_time_stamps > max_t)[0]
    indices.extend(remove_indices)

    for key in robot_keys:
        val = new_data[key]
        filtered_val = np.delete(val, indices, axis=0)
        new_data[key] = filtered_val

    return new_data


def remove_empty_data(collection: list):
    new_collection = []

    def is_empty(value):
        if isinstance(value, list) or isinstance(value, np.ndarray):
            return len(value) == 0
        return False

    for data in collection:
        add_data = True
        has_empty_entry = any(is_empty(value) for value in data.values())

        if has_empty_entry:
            add_data = False
            print("empty!")

        if add_data:
            new_collection.append(data)

    return new_collection


def remove_sparse_data(
    collection: list,
    min_num_samples: int = 100,
    max_num_samples: int = 700,
    verbose: bool = True,
):
    new_collection = []

    big_counter = 0
    small_counter = 0

    for data in collection:
        if len(data["ball_time_stamps"]) < min_num_samples:
            small_counter += 1
            continue

        if len(data["ball_time_stamps"]) > max_num_samples:
            big_counter += 1
            continue

        new_collection.append(data)

    if verbose:
        print(f"{small_counter} small trajectories removed.")
        print(f"{big_counter} many trajectories removed.")

    return new_collection


def remove_patchy_ball_data(
    collection: list, max_patch_time: float, verbose: bool = True
):
    new_collection = []

    gap_counter = 0

    for data in collection:
        time_gaps = np.diff(data["ball_time_stamps"])
        largest_gap = max(time_gaps)

        if len(time_gaps) == 0:
            print(f"Ball data is empty!")

        if largest_gap > max_patch_time:
            gap_counter += 1
            continue

        new_collection.append(data)

    if verbose:
        print(f"{gap_counter} patchy data removed.")

    return new_collection


def filter_outside_region(
    data,
    xlimit: Tuple[float],
    ylimit: Tuple[float],
    zlimit: Tuple[float],
):
    limits = [xlimit, ylimit, zlimit]

    positions = data["ball_positions"]
    delete_indices = []

    for idx in range(len(positions)):
        for axis in range(3):
            if not (limits[axis][0] < positions[idx][axis] < limits[axis][1]):
                delete_indices.append(idx)
                break

    return delete_indices


def remove_samples(item, indices_to_remove):
    filtered_item = item.copy()

    for key, values in item.items():
        if key == "launch_parameter":
            continue

        if key == "date_stamp":
            continue

        print(f"Before deletion: key='{key}', values.shape={values.shape}")
        try:
            values = np.delete(values, indices_to_remove, axis=0)
        except IndexError as e:
            print(f"Error: {e}")
            print(indices_to_remove)

        print(f"After deletion: key='{key}', values.shape={values.shape}")
        print("#################################")
        filtered_item[key] = values

    return filtered_item
