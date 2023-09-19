import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tomlkit

from ball_prediction.utils.data_management import load_robot_ball_data

FILE_PATH = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/no_spin_robot.hdf5"
INDEX = 59  # clean sample at 59


def visualize_data_3d():
    marker_size = 1.25

    # index = str(INDEX)
    file_path = FILE_PATH

    collection = load_robot_ball_data(file_path=file_path)

    ax = plt.figure().add_subplot(projection="3d")

    for item in collection:
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
