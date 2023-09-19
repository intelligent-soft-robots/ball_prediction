import pickle

import h5py
import numpy as np
from tqdm import tqdm


def load_robot_ball_data(file_path: str):
    file = h5py.File(file_path, "r")
    collection = []
    for index in list(file["originals"].keys()):
        collection.append(
            {
                "launch_parameter": np.array(
                    file["originals"][index]["launch_parameter"]
                ),
                "date_stamp": file["originals"][index]["date_stamp"],
                "time_stamps": np.array(file["originals"][index]["time_stamps"]),
                "ball_ids": np.array(file["originals"][index]["ball_ids"]),
                "ball_time_stamps": np.array(
                    file["originals"][index]["ball_time_stamps"]
                ),
                "ball_positions": np.array(file["originals"][index]["ball_positions"]),
                "ball_velocities": np.array(
                    file["originals"][index]["ball_velocities"]
                ),
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
        )

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


def load_data_tobuschat(dir_path: str, n_files: int):
    import matplotlib.pyplot as plt
    
    collection = []
    for i in tqdm(range(n_files)):
        data = {}
        
        ball_data_available = False
        robot_data_available = False
        
        content = []
        file = "estimator"
        file_path = dir_path+f"Iterator_{i}_{file}"
        try:
            with open(file=file_path, mode="rb") as file:
                while True:
                    try:
                        content.append(pickle.load(file))
                        
                    except:
                        break
                    
                data["ball_time_stamps"] = content[0][0]
                data["ball_positions"] = content[1].T
            
            ball_data_available = True
        except Exception as e:
            continue
        
        content = []   
        file = "robot"
        file_path = dir_path+f"Iterator_{i}_{file}"
        try:
            with open(file=file_path, mode="rb") as file:
                while True:
                    try:
                        content.append(pickle.load(file))
                    except:
                        break
        
                data["robot_time_stamps"] = content[2]
                data["robot_joint_angles"] = content[0]
                
            robot_data_available = True
        except Exception as e:
            continue

        if robot_data_available and ball_data_available:
            collection.append(data)
            # print(f"Data {i} added.")
            
            # print(f"Len robot: {np.array(data['robot_time_stamps']).shape}")
            # print(f"Len ball: {np.array(data['ball_time_stamps']).shape}")
            # print(np.array(data['robot_time_stamps']))
            # print(np.array(data['ball_time_stamps']))

    print(f"Length collection: {len(collection)}")
    
    return collection

def to_hdf5(file_path: str, collection: list):
    file = h5py.File(file_path, "a")
    dataset = file.create_group("originals")
    
    for id, data in enumerate(collection):
        iteration = dataset.create_group(str(id))
        
        iteration.create_dataset("ball_time_stamps", data=data["ball_time_stamps"])
        iteration.create_dataset("ball_positions", data=data["ball_positions"])
        
        iteration.create_dataset("robot_time_stamps", data=data["robot_time_stamps"])
        iteration.create_dataset("robot_joint_angles", data=data["robot_joint_angles"])
        
