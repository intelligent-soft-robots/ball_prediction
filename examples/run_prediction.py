import logging
import pathlib
import time
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
    prediction_config_path = pathlib.Path(
        "/home/adittrich/test_workspace/workspace/src/ball_prediction/config/config.toml"
    )
    predict_config = load_toml(prediction_config_path)

    return predict_config


def load_data(index: Optional[int] = None):
    # Get test data
    path = "/home/adittrich/test_workspace/workspace/src/ball_prediction/examples/data/low_spin_data.hdf5"
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


def run_predictor():
    time_stamps, positions, velocities = load_data(12)
    config = load_config()

    # Load predictor
    predictor = TrajectoryPredictor(config)

    # Load parameters
    init_buffer_size = config["setting"]["init_buffer_size"]

    # Load storage
    time_stamps_predictions = []
    position_predictions = []

    time_stamps_predictions_unfiltered = []
    position_predictions_unfiltered = []

    for i in range(len(positions)):
        t_current = time_stamps[i]
        p = positions[i]
        v = velocities[i]

        z = np.hstack((p, v))

        t_0 = time.time()
        # <------------------------------------>
        predictor.input_samples(z, t_current)
        # <------------------------------------>
        deltat = time.time() - t_0

        if i % 10 == 0 and i != 0:
            print(f"Update time: {deltat}")

            t_0 = time.time()
            # <-------------------------->
            predictor.predict_horizon()
            # <-------------------------->
            deltat = time.time() - t_0
            print(f"Prediction time: {deltat}")

            # Uniltered Predictions
            t_pred, q_pred = predictor.get_prediction(filter=False)
            time_stamps_predictions_unfiltered.append(t_pred)
            position_predictions_unfiltered.append(q_pred)

            # Filtered Predictions
            t_pred, q_pred = predictor.get_prediction()
            time_stamps_predictions.append(t_pred)
            position_predictions.append(q_pred)

    positions_estimated = np.array(predictor.q_estimated)

    _plotting_predictions(
        time_stamps[init_buffer_size:],
        positions[init_buffer_size:],
        positions_estimated,
        time_stamps_predictions,
        position_predictions,
        time_stamps_predictions_unfiltered,
        position_predictions_unfiltered,
    )


def prediction_error():
    collection = load_data()
    collection = collection[0:250]

    # Load predictor
    config = load_config()
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


def _plotting_predictions(
    t_measured,
    p_measured,
    p_estimated,
    t_predictions,
    p_predictions,
    t_predictions_unfiltered,
    p_predictions_unfiltered,
):
    t_measured = np.array(t_measured)
    p_measured = np.array(p_measured)

    p_estimated = np.array(p_estimated)

    alpha = 0.6
    colors = [
        "#46b361",
        "#3fad82",
        "#36a79b",
        "#2ca1b1",
        "#1f9ac5",
        "#0193d7",
    ]  # 6 colors

    colors = [
        "#46b361",
        "#42b074",
        "#3eac85",
        "#39a993",
        "#34a6a1",
        "#2fa2ad",
        "#289eb8",
        "#219bc3",
        "#1797cd",
        "#0193d7",
    ]  # 10 colors

    colors = [
        "#46b361",
        "#44b16b",
        "#42b073",
        "#40ae7b",
        "#3ead83",
        "#3cab8a",
        "#3aaa91",
        "#38a898",
        "#35a69e",
        "#33a5a4",
        "#30a3aa",
        "#2da1af",
        "#2aa0b5",
        "#279eba",
        "#249cbf",
        "#209ac4",
        "#1c98c9",
        "#1797ce",
        "#1095d2",
        "#0193d7",
    ]  # 20 colors

    fig, axs = plt.subplots(3, 1)

    for i, ax in enumerate(axs):
        j = 0
        for t_predicted, p_predicted in zip(
            t_predictions_unfiltered, p_predictions_unfiltered
        ):
            if len(p_predicted) != 0:
                t_predicted = np.array(t_predicted)
                p_predicted = np.array(p_predicted)

                ax.plot(
                    t_predicted,
                    p_predicted[:, i],
                    label=f"prediction {np.round(t_predicted[0],3)}",
                    color=colors[j],
                    alpha=alpha,
                )
                j += 1

        ax.plot(t_measured, p_measured[:, i], label="measured")
        ax.plot(t_measured, p_estimated[:, i], label="estimate", linestyle="dotted")

        ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    i = 0
    for t_predicted, p_predicted in zip(t_predictions, p_predictions):
        if len(p_predicted) != 0:
            t_predicted = np.array(t_predicted)
            p_predicted = np.array(p_predicted)

            ax.scatter(
                p_predicted[:, 0],
                p_predicted[:, 1],
                p_predicted[:, 2],
                label=f"Prediction {np.round(t_predicted[0],3)}",
                color=colors[i],
                alpha=alpha,
            )

            i += 1

    i = 0
    for t_predicted_un, p_predicted_un in zip(
        t_predictions_unfiltered, p_predictions_unfiltered
    ):
        if len(t_predicted_un) != 0:
            t_predicted_un = np.array(t_predicted_un)
            p_predicted_un = np.array(p_predicted_un)

            ax.plot(
                p_predicted_un[:, 0],
                p_predicted_un[:, 1],
                p_predicted_un[:, 2],
                label=f"Prediction {np.round(t_predicted_un[i],3)}",
                color=colors[i],
                alpha=alpha,
            )

            i += 1

    ax.plot(
        p_measured[:, 0],
        p_measured[:, 1],
        p_measured[:, 2],
        label="Measurements",
        linestyle="-",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.set_xlim3d([0, 3])
    ax.set_ylim3d([-1.5, 1.5])
    ax.set_zlim3d([0, 3])

    ax.legend()

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run_predictor()
    # prediction_error()
