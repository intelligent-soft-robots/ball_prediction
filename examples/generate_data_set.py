from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from ball_prediction.models.magnus_regressor import (
    PHYSICS_CFG,
    REGRESS_CFG,
    MagnusRegressor,
    compute_velocity_regression,
)
from ball_prediction.models.rebound_detection import detect_rebounds, filter_rebounds
from ball_prediction.models.spin_estimator import ContactType
from ball_prediction.utils.data_filter import (
    filter_first_n_samples,
    filter_last_n_samples,
    filter_outside_region,
    filter_time_span,
    remove_empty_data,
    remove_patchy_ball_data,
    remove_samples,
    remove_sparse_data,
)
from ball_prediction.utils.data_management import load_robot_ball_data, load_data_tobuschat

FILE_PATH = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/no_spin_robot.hdf5"


def load_dataset():
    file_path = FILE_PATH
    collection = load_robot_ball_data(file_path)
    return collection


# load dataset
def filter_dataset():
    _collection = load_large_dataset()
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
    zlimit = (-1.40, 2.0)

    new_collection = []
    for data in _collection:
        data = filter_outside_region(data, xlimit, ylimit, zlimit)
        new_collection.append(data)
    _collection = new_collection
    print(f"Region filtered data: {len(_collection)}")

    # Patchy Filter
    _collection = remove_patchy_ball_data(_collection, 0.2)
    print(f"Patch filtered data: {len(_collection)}")

    _collection = remove_sparse_data(_collection)

    # Filter last samples as they are often noisy
    new_collection = []
    for data in _collection:
        data = filter_last_n_samples(data, 70)
        new_collection.append(data)
    _collection = new_collection
    print(f"Last filtered data: {len(_collection)}")

    # Filter first samples as they are often noisy
    new_collection = []
    for data in _collection:
        # data = filter_first_n_samples(data, 30)
        new_collection.append(data)
    _collection = new_collection
    print(f"Last filtered data: {len(_collection)}")

    print(f"Filtered collection: {len(_collection)}")

    return _collection


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
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])

            for i in range(3):
                axs[i].plot(time_stamps, positions[:, i])

        except Exception as e:
            print(e)
            counter += 1
            # print(data["ball_time_stamps"])

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


def load_large_dataset():
    collection_path_1 = "/home/lis/Desktop/TTData/selected_data/contact_1.hdf5"
    collection_path_2 = "/home/lis/Desktop/TTData/selected_data/contact_2.hdf5"
    collection_path_3 = "/home/lis/Desktop/TTData/selected_data/contact_3.hdf5"
    collection_path_4 = "/home/lis/Desktop/TTData/selected_data/contact_4.hdf5"
    collection_path_5 = "/home/lis/Desktop/TTData/selected_data/contact_5.hdf5"
    collection_path_6 = "/home/lis/Desktop/TTData/selected_data/contact_6.hdf5"

    file_paths = [
        collection_path_1,
        collection_path_2,
        collection_path_3,
        collection_path_4,
        collection_path_5,
        collection_path_6,
    ]

    collection = []
    for path in file_paths:
        _collection = load_robot_ball_data(path)
        collection.extend(_collection)

    print(len(collection))

    return collection


def export_dataset():
    # store dataset
    ...


def generate_dataset_tobuschat():
    file_path = "/home/lis/Desktop/TTData/recording_philipp_tobuschat/processed_data/Data/"
    n_files = 13000
    
    collection = load_data_tobuschat(file_path, n_files)
    
    # export_path = "/home/lis/Desktop/TTData/recording_philipp_tobuschat/contact_tobuschat.hdf5"
    # to_hdf5(export_path, collection)
    
    # data = collection[900]
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # positions = np.array(data["ball_positions"])
    return collection


def split_dataset():
    idx_delta_before = 4
    idx_delta_after = 0
    idx_delta_racket = -2

    num_reg_samples = 30
    index_threshold = (
        num_reg_samples  # ensures enough regression samples between rebounds
    )
    poly_deg = 3

    magnus_regressor = MagnusRegressor(
        regression_config=REGRESS_CFG, physics_config=PHYSICS_CFG
    )

    filtered_collection = filter_dataset()

    for data in filtered_collection:
        ball_time_stamps = data["ball_time_stamps"]
        ball_positions = data["ball_positions"]
        ball_velocities = data["ball_velocities"]

        contact_dict = detect_rebounds(
            time_stamps=ball_time_stamps, positions=ball_positions
        )

        contact_dict = filter_rebounds(
            contact_dict=contact_dict,
            time_stamps=ball_time_stamps,
            index_threshold=index_threshold,
        )

        for key, value in contact_dict.items():
            idx_before = key - idx_delta_before
            vel_x, vel_y, vel_z = ball_velocities[idx_before]

            ts = ball_time_stamps[idx_before - num_reg_samples : idx_before]
            ps = ball_positions[idx_before - num_reg_samples : idx_before]

            regressed_velocity = compute_velocity_regression(ts, ps, poly_deg)
            vel_x_after, vel_y_after, vel_z_after = regressed_velocity[0]

            magnus_regressor.compute(
                ts,
                ps,
            )


def count_rebounds(collection):
    print("Start counting!")
    table_rebounds = 0
    racket_rebounds = 0

    index_threshold = 30

    for data in tqdm.tqdm(collection):
        ball_time_stamps = data["ball_time_stamps"]
        ball_positions = data["ball_positions"]

        contact_dict = detect_rebounds(
            time_stamps=ball_time_stamps, positions=ball_positions
        )
        contact_dict = filter_rebounds(
            contact_dict=contact_dict,
            time_stamps=ball_time_stamps,
            index_threshold=index_threshold,
        )

        for key, value in contact_dict.items():
            if value == ContactType.RACKET:
                racket_rebounds += 1

            if value == ContactType.TABLE:
                table_rebounds += 1

    print(f"Table rebounds: {table_rebounds}")
    print(f"Racket rebounds: {racket_rebounds}")


if __name__ == "__main__":
    #filtered_collection = filter_dataset()
    # count_rebounds(filtered_collection)
    filtered_collection = generate_dataset_tobuschat()
    

    # Visualize
    plot_ball_data(filtered_collection)
    plt.show()
