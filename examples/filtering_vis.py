import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tomlkit

FILE_PATH = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/examples/data/ball_trajectories_with_return.hdf5"
INDEX = 59  # clean sample at 59


def load_toml(file_path: str):
    with open(pathlib.Path(file_path), mode="r") as fp:
        config = fp.read()
        config_dict = dict(tomlkit.parse(config))

    return config_dict


def load_data():
    file = h5py.File(FILE_PATH, "r")
    collection = {}
    for index in list(file["originals"].keys()):
        collection[index] = {
            "launch_parameter": np.array(file["originals"][index]["launch_parameter"]),
            "date_stamp": file["originals"][index]["date_stamp"],
            "time_stamps": np.array(file["originals"][index]["time_stamps"]),
            "ball_ids": np.array(file["originals"][index]["ball_ids"]),
            "ball_time_stamps": np.array(file["originals"][index]["ball_time_stamps"]),
            "ball_positions": np.array(file["originals"][index]["ball_positions"]),
            "ball_velocities": np.array(file["originals"][index]["ball_velocities"]),
            "robot_time_stamps": np.array(
                file["originals"][index]["robot_time_stamps"]
            ),
            "robot_joint_angles": np.array(
                file["originals"][index]["robot_joint_angles"]
            ),
            "robot_joint_angles_desired": np.array(
                file["originals"][index]["robot_joint_angles_desired"]
            ),
            "robot_joint_angle_velocities": np.array(
                file["originals"][index]["robot_joint_angle_velocities"]
            ),
            "robot_joint_angle_velocities_desired": np.array(
                file["originals"][index]["robot_joint_angle_velocities_desired"]
            ),
            "robot_pressures_agonist": np.array(
                file["originals"][index]["robot_pressures_agonist"]
            ),
            "robot_pressures_antagonist": np.array(
                file["originals"][index]["robot_pressures_antagonist"]
            ),
        }

    return collection


def visualize_data_3d():
    marker_size = 1.25
    # index = str(INDEX)

    collection = load_data()

    ax = plt.figure().add_subplot(projection="3d")

    for key, item in collection.items():
        ball_time_stamps = item["ball_time_stamps"]
        ball_positions = item["ball_positions"]
        ball_velocities = item["ball_velocities"]

        ball_positions = np.array(ball_positions)[10:]

        try:
            ax.plot(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2])
        except:
            pass

    print(ball_positions)
    ax.set_aspect("equal")


if __name__ == "__main__":
    visualize_data_3d()

    plt.show()
