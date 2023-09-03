from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ball_prediction.utils.data_filter import (
    filter_outside_region,
    filter_time_span,
    remove_empty_data,
    remove_patchy_ball_data,
    remove_samples,
    remove_sparse_data,
)
from ball_prediction.utils.data_management import load_robot_ball_data

FILE_PATH = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/no_spin_robot.hdf5"


def load_dataset():
    file_path = FILE_PATH
    collection = load_robot_ball_data(file_path)
    return collection


# load dataset
def filter_dataset():
    _collection = load_dataset()
    print(f"Raw data: {len(_collection)}")

    # Data Length Filter
    _collection = remove_sparse_data(_collection)

    # Time Span Filter
    min_time_step = 3.0
    max_time_step = 4.5

    new_collection = []
    for data in _collection:
        data = filter_time_span(data, min_time_step, max_time_step)
        new_collection.append(data)
    _collection = new_collection
    print(f"Time span filtered data: {len(_collection)}")

    # Region Filter
    xlimit = (-3.0, 3.0)
    ylimit = (-3.0, 1.0)
    zlimit = (-1.4, 2.0)

    new_collection = []
    for data in _collection:
        data = filter_outside_region(data, xlimit, ylimit, zlimit)
        new_collection.append(data)
    _collection = new_collection
    print(f"Region filtered data: {len(_collection)}")

    # Patchy Filter
    _collection = remove_patchy_ball_data(_collection, 0.2)
    print(f"Patch filtered data: {len(_collection)}")

    # Visualize
    plot_ball_data(_collection)


def plot_ball_data(collection: list):
    ax = plt.figure().add_subplot(projection="3d")
    fig, axs = plt.subplots(3)

    counter = 0
    for data in collection:
        ball_time_stamps = data["ball_time_stamps"]
        ball_positions = data["ball_positions"]

        time_stamps = np.array(ball_time_stamps)
        positions = np.array(ball_positions)

        try:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])

            for i in range(3):
                axs[i].plot(time_stamps, positions[:, i])

        except Exception as e:
            counter += 1
            print(data["ball_time_stamps"])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_aspect("equal")

    print(f"{counter} data not plotted.")


def plot_all_data(collection):
    fig, axs = plt.subplots(len(data))
    ax = plt.figure().add_subplot(projection="3d")
    counter = 0

    data = collection[0]

    for i, (key, val) in enumerate(data.items()):
        if key == "launch_parameter":
            print(val)

        if key == "time_stamps":
            print(val)

        if key == "ball_time_stamps":
            print(val)

        if key == "robot_time_stamps":
            print(val)

        axs[i].plot(val)
        axs[i].set_ylabel(key)

    plt.show()


def export_dataset():
    # store dataset
    ...


def split_dataset():
    # store dataset
    ...


if __name__ == "__main__":
    filter_dataset()

    plt.show()
