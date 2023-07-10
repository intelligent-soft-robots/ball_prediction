import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import signal_handler
import tomlkit

from ball_prediction.trajectory_prediction import TrajectoryPredictor


def configure():
    global TENNICAM_CLIENT_DEFAULT_SEGMENT_ID
    global TENNICAM_CLIENT_DEFAULT_FREQUENCY
    config = BrightArgs("o80 tennicam client standalone")
    config.add_option(
        "segment_id",
        TENNICAM_CLIENT_DEFAULT_SEGMENT_ID,
        "segment_id of the o80 backend",
        str,
    )
    config.add_option(
        "config_path",
        tennicam_client.get_default_config_file(),
        "configuration file",
        str,
        [FileExists()],
    )
    config.add_option(
        "frequency", TENNICAM_CLIENT_DEFAULT_FREQUENCY, float, [Positive()]
    )
    config.add_option(
        "active_transform",
        False,
        "if true, the driver will read transform parameter at each iteration",
        bool,
    )
    change_all = False
    config.dialog(change_all, sys.argv[1:])
    print()
    return config


def _load_toml(file_path: str):
    with open(Path(file_path), mode="r") as fp:
        config = fp.read()
        config_dict = dict(tomlkit.parse(config))

    return config_dict


class TennicamClientPredictor:
    def __init__(self, config_path) -> None:
        config = _load_toml(config_path)
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


def _logging(argv):
    # the first argument is the script name, so we skip that
    if len(argv) < 2:
        print(
            "No output file path provided. Please provide a path for the output file."
        )
        path = Path(input())
        while not path.parent.exists():
            print("Invalid path, directory doesn't exist. Please enter a valid path:")
            path = Path(input())
    else:
        path = Path(argv[1])
        if not path.parent.exists():
            print(f"Destination directory {path.parent} not found.")
            return -1

    script_dir = Path(__file__).resolve().parent
    config_dir = script_dir.parent / "config"
    path = config_dir / "config.toml"

    predictor = TennicamClientPredictor(path)

    try:
        with h5py.File(str(path), "a") as f:
            # Use the number of existing groups as the counter
            counter = len(f.keys())

            while True:
                # Collect some data
                time_stamps = np.random.random((10, 3))
                positions = np.random.random((10, 3))
                velocities = np.random.random((10, 3))
                launch_parameters = np.random.random((10, 3))

                # Create a new group for this set of data
                grp = f.create_group(f"data_{counter}")

                # Add the datasets to the group
                grp.create_dataset("timestamps", data=time_stamps)
                grp.create_dataset("positions", data=positions)
                grp.create_dataset("velocities", data=velocities)
                grp.create_dataset("launch_parameters", data=launch_parameters)

                counter += 1

    except KeyboardInterrupt:
        print("Data collection stopped by user.")


if __name__ == "__main__":
    _logging(sys.argv)

    config = configure()
    run(
        config.segment_id, config.frequency, config.config_path, config.active_transform
    )
