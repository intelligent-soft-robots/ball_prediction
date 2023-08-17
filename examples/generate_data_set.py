from typing import Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

from ball_prediction.utils.data_import import load_robot_ball_data

FILE_PATH = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/no_spin_robot.hdf5"


def concatenate_dicts():
    file_paths = [
        "",
        "",
    ]

    concatenate_collection = {}
    for path in file_paths:
        collection = load_robot_ball_data(path)
        concatenate_collection.update(collection)

    file_name = "robot_ball_data"
    export_path = "/home/lis/switchdrive/80_data/table_tennis/"
    file_path = export_path / (file_name + ".hdf5")
    file = h5py.File(file_path, "a")

    if "originals" not in file.keys():
        dataset = file.create_group("originals")
    else:
        dataset = file["originals"]

    if list(dataset.keys()):
        id = max([int(i) for i in dataset.keys()]) + 1
    else:
        id = 0

    iteration = dataset.create_group(str(id))

    # create data
    iteration.create_dataset("launch_parameter", data=launch_parameter)
    iteration.create_dataset("date_stamp", data=str(now.strftime("%y%m%d%H%M%s")))
    iteration.create_dataset("time_stamps", data=time_stamps)

    iteration.create_dataset("ball_ids", data=ball_ids)
    iteration.create_dataset("ball_time_stamps", data=ball_time_stamps)
    iteration.create_dataset("ball_positions", data=ball_positions)
    iteration.create_dataset("ball_velocities", data=ball_velocities)

    iteration.create_dataset("robot_time_stamps", data=robot_time_stamps)
    iteration.create_dataset("robot_joint_angles", data=robot_joint_angles)
    iteration.create_dataset(
        "robot_joint_angles_desired", data=robot_joint_angles_desired
    )
    iteration.create_dataset(
        "robot_joint_angle_velocities", data=robot_joint_angle_velocities
    )
    iteration.create_dataset(
        "robot_joint_angle_velocities_desired",
        data=robot_joint_angle_velocities_desired,
    )
    iteration.create_dataset("robot_pressures_agonist", data=robot_pressures_agonist)
    iteration.create_dataset(
        "robot_pressures_antagonist", data=robot_pressures_antagonist
    )


def remove_indices_from_collection(collection, indices_to_remove):
    for index, data in collection.items():
        for key in data:
            if isinstance(data[key], np.ndarray):
                collection[index][key] = np.delete(data[key], indices_to_remove, axis=0)

    return collection


# load dataset
def loading():
    marker_size = 1.25

    file_path = FILE_PATH

    collection = load_robot_ball_data(file_path)

    ax = plt.figure().add_subplot(projection="3d")
    counter = 0

    min_time_step = 3.0
    max_time_step = 4.5
    
    x_lims = (-3.0, 3.0)
    y_lims = (-3.0, 3.0)
    z_lims = (-1.4, 2.0)
    
    fig, axs = plt.subplots(3)

    for key, item in collection.items():
        ball_time_stamps = item["ball_time_stamps"]
        ball_positions = item["ball_positions"]
        ball_velocities = item["ball_velocities"]

        time_stamps = np.array(ball_time_stamps)
        positions = np.array(ball_positions)
        velocities = np.array(ball_velocities)

        indices = filter_time_span(item, min_time_step, max_time_step)
        indices = filter_outside_region(item, x_lims, y_lims, z_lims)

        try:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])

            for i in range(3):
                axs[i].plot(time_stamps, positions[:, i])

        except:
            counter += 1

    print(counter)


def filter_time_span(item, min_t, max_t):
    time_stamps = item["ball_time_stamps"]

    indices = []
    remove_indices = np.nonzero(min_t > time_stamps)[0]
    indices.extend(remove_indices)

    remove_indices = np.nonzero(time_stamps > max_t)[0]
    indices.extend(remove_indices)

    print(indices)
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


def export_dataset():
    # store dataset
    ...


def split_dataset():
    ...


if __name__ == "__main__":
    loading()

    plt.show()
