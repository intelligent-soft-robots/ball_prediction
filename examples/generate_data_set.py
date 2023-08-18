from typing import Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

from ball_prediction.utils.data_management import load_robot_ball_data
from ball_prediction.utils.data_filter import filter_time_span, filter_outside_region, remove_samples

FILE_PATH = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/no_spin_robot.hdf5"


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

        removal_indices = []
        
        indices = filter_time_span(item, min_time_step, max_time_step)
        removal_indices.extend(indices)
        
        #indices = filter_outside_region(item, x_lims, y_lims, z_lims)
        #removal_indices.extend(indices)
        
        item = remove_samples(item, removal_indices)

        try:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])

            for i in range(3):
                axs[i].plot(time_stamps, positions[:, i])

        except:
            counter += 1

    print(counter)



def export_dataset():
    # store dataset
    ...


def split_dataset():
    ...


if __name__ == "__main__":
    loading()

    plt.show()
