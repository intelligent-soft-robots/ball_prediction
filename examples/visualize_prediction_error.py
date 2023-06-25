import logging
import pathlib
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tomlkit

from ball_prediction.prediction_filter import VirtualPlane
from ball_prediction.trajectory_prediction import TrajectoryPredictor


def load_toml(file_path: str):
    with open(pathlib.Path(file_path), mode="r") as fp:
        config = fp.read()
        config_dict = dict(tomlkit.parse(config))

    return config_dict


def load_config():
    script_dir = pathlib.Path(__file__).resolve().parent
    config_dir = script_dir.parent / "config"
    prediction_config_path = config_dir / "config.toml"

    predict_config = load_toml(prediction_config_path)

    return predict_config


def load_data(index: Optional[int] = None):
    # Get test data
    script_dir = pathlib.Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    path = data_dir / "simple_ball_trajectories.hdf5"

    group = "originals"

    trajectory_collection = []
    file = h5py.File(path, "r")

    for i in list(file[group].keys()):
        trajectory_data = {}

        trajectory_data["launch_param"] = tuple(file[group][i]["launch_param"])
        trajectory_data["time_stamps"] = list(file[group][i]["time_stamps"])
        trajectory_data["positions"] = [
            tuple(entry) for entry in list(file[group][i]["positions"])
        ]
        trajectory_data["velocities"] = [
            tuple(entry) for entry in list(file[group][i]["velocities"])
        ]

        trajectory_collection.append(trajectory_data)

    file.close()

    if index is None:
        return trajectory_collection

    data = trajectory_collection[index]

    time_stamps = np.array(data["time_stamps"])
    positions = np.array(data["positions"])
    velocities = np.array(data["velocities"])

    return time_stamps, positions, velocities


def prediction_error():
    collection = load_data()
    collection = collection[0:250]

    # Load predictor
    config = load_config()
    config["filter_method"] = "plane"
    config["virtual_plane"]["axis"] = 0
    config["virtual_plane"]["offset"] = 2.0

    predictor = TrajectoryPredictor(config)
    filter = VirtualPlane(config)

    t_storage = []
    q_error_storage = []
    q_error_mse_storage = []

    for data in collection:
        time_stamps = np.array(data["time_stamps"])
        positions = np.array(data["positions"])
        velocities = np.array(data["velocities"])

        z = np.hstack((positions, velocities))

        # Logging
        time_stamps_predictions = []
        position_predictions = []

        t_ground, q_ground = filter.filter(time_stamps, z)

        t_error = []
        q_error = []
        q_error_mse = []

        index = 0
        for t, p, v in zip(time_stamps, positions, velocities):
            z = np.hstack((p, v))

            predictor.input_samples(z, t)
            t_pred, q_pred = predictor.get_prediction()

            if q_pred and q_ground:
                time_stamps_predictions.append(t_pred)
                position_predictions.append(q_pred)

                error = q_pred[0] - q_ground[0]
                # print(f"Ground: {q_ground}, Prediction: {q_pred}, Error: {error} ")

                t_error.append(t)
                q_error.append(error)
                q_error_mse.append(np.linalg.norm(error))

        index += 1

        t_storage.append(t_error)
        q_error_storage.append(q_error)
        q_error_mse_storage.append(q_error_mse)

        predictor.reset_ekf()

    fig, axs = plt.subplots(4, 1, sharex=True)

    for t_error, q_error, q_error_mse in zip(
        t_storage, q_error_storage, q_error_mse_storage
    ):
        q_error = np.array(q_error)
        for i in range(3):
            if len(q_error) != 0:
                axs[i].plot(t_error, q_error[:, i])

        axs[3].scatter(t_error, q_error_mse)

    axs[0].set_ylabel(r"$\Delta p$ [m]")
    axs[1].set_ylabel(r"$\Delta p$ [m]")
    axs[2].set_ylabel(r"$\Delta p$ [m]")
    axs[3].set_ylabel(r"$MSE(\Delta p)$ [-]")

    axs[3].set_xlabel("t [s]")

    for ax in axs:
        ax.set_ylim((-0.1, 0.8))

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prediction_error()
