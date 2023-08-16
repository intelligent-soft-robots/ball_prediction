import pathlib
import h5py
import tomlkit
import numpy as np
import matplotlib.pyplot as plt

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


def compute_magnus():
    marker_size = 1.25
    index = str(INDEX)
    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    start = 10
    end = 60

    time_stamps = ball_time_stamps[start:end]
    positions = ball_positions[start:end]
    velocities = ball_velocities[start:end]

    poly_deg = 5

    physics_config = {
        "ball_mass": 0.027,
        "ball_radius": 0.02,
        "air_density": 1.18,
        "gravitational_constant": 9.80801,
        "drag_coefficient": 0.47016899,
        "lift_coefficient": 1.46968343,
        "decay_coefficient": 0.005054473513775579,
    }

    m_ball = physics_config["ball_mass"]
    r_ball = physics_config["ball_radius"]
    rho = physics_config["air_density"]
    g = physics_config["gravitational_constant"]
    c_drag = physics_config["drag_coefficient"]
    c_lift = physics_config["lift_coefficient"]
    c_decay = physics_config["decay_coefficient"]

    A = np.pi * r_ball**2

    v = []
    a = []

    for axis in range(3):
        poly = np.polynomial.polynomial.Polynomial.fit(
            time_stamps, positions[:, axis], deg=poly_deg
        )
        v_poly = poly.deriv()
        a_poly = poly.deriv()

        v_axis = []
        a_axis = []
        for t in time_stamps:
            v_axis.append(v_poly(t))
            a_axis.append(a_poly(t))

        v.append(v_axis)
        a.append(a_axis)

    v = np.array(v)
    a = np.array(a)

    v = np.concatenate(
        (v[0].reshape(-1, 1), v[1].reshape(-1, 1), v[2].reshape(-1, 1)), axis=1
    )

    a = np.concatenate(
        (a[0].reshape(-1, 1), a[1].reshape(-1, 1), a[2].reshape(-1, 1)), axis=1
    )

    F_ball = m_ball * a

    F_g = -g * np.ones_like(a)
    F_g[:, :2] = 0

    F_drag = -0.5 * rho * c_drag * A / m_ball * np.linalg.norm(v) * v

    F_lift = F_ball - F_g - F_drag

    fig, axs = plt.subplots(9)
    for i in range(3):
        axs[i].scatter(time_stamps, velocities[:, i], s=marker_size)

        # Regressed
        axs[i].plot(time_stamps, v[:, i])
        axs[i + 3].plot(time_stamps, a[:, i])

        # Computed
        axs[i + 6].plot(time_stamps, F_g[:, i], label="gravity")
        axs[i + 6].plot(time_stamps, F_drag[:, i], label="drag")
        axs[i + 6].plot(time_stamps, F_ball[:, i], label="total")
        axs[i + 6].plot(time_stamps, F_lift[:, i], label="magnus")

        axs[i + 6].legend()


def magnus_regressor_test():
    from ball_prediction.ball_prediction.contact_models.magnus_regressor import (
        MagnusRegressor,
        REGRESS_CFG,
        PHYSICS_CFG,
    )

    regressor = MagnusRegressor(REGRESS_CFG, PHYSICS_CFG)

    index = str(INDEX)
    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    start = 10
    end = 40

    time_stamps = np.array(ball_time_stamps)[start:end]
    positions = np.array(ball_positions)[start:end]

    eval_time = ball_time_stamps[40]

    F_lift = regressor.compute(time_stamps, positions, eval_time)

    print(F_lift)


if __name__ == "__main__":
    magnus_regressor_test()

    plt.show()
