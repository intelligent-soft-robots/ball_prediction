import h5py
import numpy as np


def load_robot_ball_data(file_path: str):
    file = h5py.File(file_path, "r")
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


def load_ball_data(file_path: str):
    file = h5py.File(file_path, "r")
    collection = {}

    for index in list(file["originals"].keys()):
        collection[index] = {
            "launch_parameters": np.array(file["originals"][index]["launch_param"]),
            "ball_time_stamps": np.array(file["originals"][index]["time_stamps"]),
            "ball_positions": np.array(file["originals"][index]["positions"]),
            "ball_velocities": np.array(file["originals"][index]["velocities"]),
        }

    return collection


def load_robot_ball_data_tobuschat(file_path: str):
    pass


def concatenate_dicts():
    file_paths = [
        "",
        "",
    ]

    concatenate_collection = {}
    for path in file_paths:
        collection = load_robot_ball_data(path)
        concatenate_collection.update(collection)

