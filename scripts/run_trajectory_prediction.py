import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tomlkit

from aimy_target_shooting.export_tools import import_all_from_hdf5
from ball_prediction.trajectory_prediction import TrajectoryPredictor
from .utils import load_toml

def load_predictor():
    prediction_config_path = pathlib.Path("./config/prediction.toml")
    predict_config = load_toml(prediction_config_path)

    return TrajectoryPredictor(predict_config)

def run_predictor():
    # Get test data
    path = "/home/adittrich/Nextcloud/82_Data_Processed/MN5008_training_data_with_outlier/MN5008_grid_data_equal_speeds.hdf5"
    collection = import_all_from_hdf5(file_path=path)

    data = collection.get_item(12)
    time_stamps = np.array(data.time_stamps)
    positions = np.array(data.positions)
    velocities = np.array(data.velocities)

    # Init param Kalman
    q_init = [0.0, 0.0, 2.0, 1.2, 0.5, 1.2, 0.0, 0.0, 0.0]

    q_init[0:3] = positions[9, :]
    q_init[3:6] = velocities[9, :]

    P_init = np.eye(9)
    t_init = time_stamps[9]

    # Load predictor
    predictor = load_predictor()
    predictor.initialize_predictor(time_stamp=t_init, q_init=q_init, P_init=P_init)

    time_stamps_predictions = []
    position_predictions = []

    time_stamps_predictions_unfiltered = []
    position_predictions_unfiltered = []

    for i in range(10, len(positions)):
        t_current = time_stamps[i]
        p = positions[i]
        v = velocities[i]

        z = np.hstack((p, v))

        t_0 = time.time()
        predictor.kalman_update_step(z, t_current)
        dt = time.time() - t_0

        if i % 15 == 0:
            # print(f"Update time: {dt}")

            t_0 = time.time()
            # <-------------------------->
            predictor.predict_horizont()
            # <-------------------------->
            dt = time.time() - t_0
            print(f"Prediction time: {dt}")

            t_pred, q_pred = predictor.get_prediction(filter=False)

            time_stamps_predictions_unfiltered.append(t_pred)
            position_predictions_unfiltered.append(q_pred)

            t_pred, q_pred = predictor.get_prediction()

            time_stamps_predictions.append(t_pred)
            position_predictions.append(q_pred)

    t_ests = np.array(predictor.t_ests)
    q_ests = np.array(predictor.q_ests)

    _plotting_predictions(
        time_stamps,
        positions,
        t_ests,
        q_ests,
        time_stamps_predictions,
        position_predictions,
        time_stamps_predictions_unfiltered,
        position_predictions_unfiltered,
    )


def run_predictor_with_initial_state_estimator():
    path = "/home/adittrich/Nextcloud/82_Data_Processed/MN5008_training_data_with_outlier/MN5008_grid_data_equal_speeds.hdf5"
    collection = import_all_from_hdf5(file_path=path)

    data = collection.get_item(12)
    time_stamps = np.array(data.time_stamps)
    positions = np.array(data.positions)
    velocities = np.array(data.velocities)

    # Load predictor
    predictor = load_predictor()

    # Logging
    time_stamps_predictions = []
    position_predictions = []

    time_stamps_predictions_unfiltered = []
    position_predictions_unfiltered = []

    for i in range(len(positions)):
        t_current = time_stamps[i]
        p = positions[i]
        v = velocities[i]

        z = np.hstack((p, v))

        predictor.input_samples(z, t_current)

        if i % 15 == 0:
            t_pred, q_pred = predictor.get_prediction(filter=False)

            if q_pred is not None:
                time_stamps_predictions_unfiltered.append(t_pred)
                position_predictions_unfiltered.append(q_pred)

            t_pred, q_pred = predictor.get_prediction()

            if q_pred is not None:
                time_stamps_predictions.append(t_pred)
                position_predictions.append(q_pred)

    t_ests = np.array(predictor.t_ests)
    q_ests = np.array(predictor.q_ests)

    _plotting_predictions(
        time_stamps,
        positions,
        t_ests,
        q_ests,
        time_stamps_predictions,
        position_predictions,
        time_stamps_predictions_unfiltered,
        position_predictions_unfiltered,
    )


def prediction_error():
    path = "/home/adittrich/Nextcloud/82_Data_Processed/MN5008_training_data_with_outlier/MN5008_grid_data_equal_speeds.hdf5"
    collection = import_all_from_hdf5(file_path=path)
    collection = collection[0:40]

    # Load predictor
    predictor = load_predictor()

    t_storage = []
    q_error_storage = []
    q_error_mse_storage = []

    for data in collection:
        time_stamps = np.array(data.time_stamps)
        positions = np.array(data.positions)
        velocities = np.array(data.velocities)

        z = np.hstack((positions, velocities))

        # Logging
        time_stamps_predictions = []
        position_predictions = []

        t_ground, q_ground = filter.filter(time_stamps, z)

        t_error = []
        q_error = []
        q_error_mse = []

        for i in range(len(positions)):
            t_current = time_stamps[i]
            p = positions[i]
            v = velocities[i]

            z = np.hstack((p, v))

            predictor.input_samples(z, t_current)
            t_pred, q_pred = predictor.get_prediction()

            if q_pred and q_ground:
                time_stamps_predictions.append(t_pred)
                position_predictions.append(q_pred)

                error = q_pred[0] - q_ground[0]
                # print(f"Ground: {q_ground}, Prediction: {q_pred}, Error: {mse} ")

                t_error.append(t_current)
                q_error.append(error)
                q_error_mse.append(np.linalg.norm(error))

        t_storage.append(t_error)
        q_error_storage.append(q_error)
        q_error_mse_storage.append(q_error_mse)

    fig, axs = plt.subplots(4, 1, sharex=True)

    for t_error, q_error, q_error_mse in zip(
        t_storage, q_error_storage, q_error_mse_storage
    ):
        q_error = np.array(q_error)
        for i in range(3):
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


def _plotting_prediction(t_measured, p_measured, t_predicted, q_predicted):
    t_measured = np.array(t_measured)
    p_measured = np.array(p_measured)

    t_predicted = np.array(t_predicted)
    q_predicted = np.array(q_predicted)

    fig, axs = plt.subplots(3, 1)

    for i, ax in enumerate(axs):
        ax.plot(t_measured, p_measured[:, i], label="measured")

        ax.plot(t_predicted, q_predicted[:, i], label="predicted")

        ax.legend()

    plt.show()


def _plotting_predictions(
    t_measured,
    p_measured,
    t_estimated,
    p_estimated,
    t_predictions,
    p_predictions,
    t_predictions_unfiltered,
    p_predictions_unfiltered,
):
    t_measured = np.array(t_measured)
    p_measured = np.array(p_measured)

    t_estimated = np.array(t_estimated)
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
        ax.plot(t_estimated, p_estimated[:, i], label="estimate", linestyle="dotted")

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


def test_time_prediction():
    # Get test data
    path = "/home/adittrich/Nextcloud/82_Data_Processed/MN5008_training_data_with_outlier/MN5008_grid_data_equal_speeds.hdf5"
    collection = import_all_from_hdf5(file_path=path)

    data = collection.get_item(4)
    time_stamps = np.array(data.time_stamps)
    positions = np.array(data.positions)

    q_init = [0.0, 0.0, 2.0, 1.2, 0.5, 1.2, 0.0, 0.0, 0.0]
    P_init = np.eye(9)

    # Load predictor
    predictor = load_predictor()
    predictor.initialize_kalman(q_init=q_init, P_init=P_init)

    for i in range(10, 50):
        z = positions[i]
        t_current = time_stamps[i]

        predictor.kalman_update_step(z, t_current)

    predictor.profile_predict()
    predictor.profile_update(positions[i + 1], time_stamps[i + 1])


if __name__ == "__main__":
    # prediction_error()
    run_predictor()
    # test_time_prediction()
    # run_predictor_with_initial_state_estimator()
