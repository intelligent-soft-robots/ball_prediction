import pathlib
from typing import Optional

import h5py
import numpy as np
import tomlkit

from ball_prediction.trajectory_prediction import TrajectoryPredictor

TENNICAM_CLIENT_DEFAULT_SEGMENT_ID = "tennicam_client"


def load_toml(file_path: str):
    with open(pathlib.Path(file_path), mode="r") as fp:
        config = fp.read()
        config_dict = dict(tomlkit.parse(config))

    return config_dict


def run_predictor():
    script_dir = pathlib.Path(__file__).resolve().parent
    config_dir = script_dir.parent / "config"
    path = config_dir / "config.toml"

    predictor = TennicamClientPredictor(path)

    predictor.run_predictor()


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


def generate_noisy_trajectory_streams():
    number_trajectories = 10
    number_outliers = 21
    time_stamp_noise = 0.01

    ball_id_stream = []
    time_stamps_stream = []
    positions_stream = []
    velocities_stream = []

    outlier_position = [0.202, -3.123, -4.23]
    outlier_velocity = [0.0, 0.0, 0.0]

    collection = load_data()

    dt = 0.005
    iteration = 0

    for data in collection[0:number_trajectories]:
        time_stamps = data["time_stamps"]
        positions = data["positions"]
        velocities = data["velocities"]

        outlier_indices = np.random.randint(0, len(time_stamps), number_outliers)

        index = 0
        for position, velocity in zip(positions, velocities):
            if index in outlier_indices:
                n_konsective_outliers = np.random.randint(0, 5)

                for _ in range(n_konsective_outliers):
                    ball_id_stream.append(-1)
                    time_stamps_stream.append(
                        iteration * dt + time_stamp_noise * np.random.standard_normal()
                    )
                    positions_stream.append(outlier_position)
                    velocities_stream.append(outlier_velocity)

                    iteration += 1

            ball_id_stream.append(iteration)
            time_stamps_stream.append(
                iteration * dt + time_stamp_noise * np.random.standard_normal()
            )
            positions_stream.append(position)
            velocities_stream.append(velocity)

            iteration += 1
            index += 1

        n_gap_samples = np.random.randint(60, 300)

        for _ in range(n_gap_samples):
            ball_id_stream.append(-1)
            time_stamps_stream.append(
                iteration * dt + time_stamp_noise * np.random.standard_normal()
            )
            positions_stream.append(outlier_position)
            velocities_stream.append(outlier_velocity)

            iteration += 1

    return ball_id_stream, time_stamps_stream, positions_stream, velocities_stream


def print_predictor():
    for b, t, p, v in zip(*generate_noisy_trajectory_streams()):
        print(f"{b}, {t}: {p}")


class TennicamClientPredictor:
    def __init__(self, config_path) -> None:
        config = load_toml(config_path)
        self.predictor = TrajectoryPredictor(config)

        # Predictor initialisation
        self.negative_ball_threshold = config["misc"]["n_negative_ball_threshold"]
        self.n_negative_ball_id = 0

    def run_predictor(self):
        predictions = []
        try:
            for b, t, p, v in zip(*generate_noisy_trajectory_streams()):
                if b != -1:
                    time_stamp = t
                    position = p
                    velocity = v
                    z = np.hstack((position, velocity))

                    self.predictor.input_samples(z, time_stamp)
                    self.predictor.predict_horizon()
                    prediction = self.predictor.get_prediction()
                    predictions.append(prediction)

                    print(f"{b}, Position: {position}, Predict: {len(prediction[1])}")

                    if len(prediction[1]) != 0:
                        print(prediction[1][-1])
                if b == -1:
                    self.n_negative_ball_id += 1

                    if self.n_negative_ball_id == self.negative_ball_threshold:
                        self.n_negative_ball_id = 0
                        self.predictor.reset_ekf()

        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    run_predictor()
