import matplotlib.pyplot as plt
import numpy as np

from ball_prediction.models.magnus_regressor import compute_velocity_regression
from ball_prediction.models.racket_kinematics import compute_racket_orientation
from ball_prediction.models.rebound_detection import detect_rebounds
from ball_prediction.models.spin_estimator import ContactType
from ball_prediction.utils.data_management import load_robot_ball_data

FILE_PATH = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/no_spin_robot.hdf5"
INDEX = 112  # clean sample at 59


def visualize_regressed_velocity_vectors():
    marker_size = 1.25
    index = str(INDEX)
    file_path = FILE_PATH

    regression = True
    num_reg_samples = 10
    poly_deg = 1

    arrow_length = 0.005

    idx_delta_before = 4
    idx_delta_after = 0
    idx_delta_racket = -2

    collection = load_robot_ball_data(file_path)

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    robot_joint_angles = collection[index]["robot_joint_angles"]
    robot_joint_angle_velocities = collection[index]["robot_joint_angle_velocities"]

    contact_dict, info = detect_rebounds(
        time_stamps=ball_time_stamps, positions=ball_positions, return_states=True
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        ball_positions[:, 0],
        ball_positions[:, 1],
        ball_positions[:, 2],
        s=marker_size,
        alpha=0.5,
    )

    for key, value in contact_dict.items():
        idx_before = key - idx_delta_before
        pos_x, pos_y, pos_z = ball_positions[idx_before]

        vel_x, vel_y, vel_z = ball_velocities[idx_before]
        v_mag = np.linalg.norm(ball_velocities[idx_before])

        if regression:
            ts = ball_time_stamps[idx_before - num_reg_samples : idx_before]
            ps = ball_positions[idx_before - num_reg_samples : idx_before]

            regressed_velocity = compute_velocity_regression(ts, ps, poly_deg)
            vel_x, vel_y, vel_z = regressed_velocity[0]
            v_mag = np.linalg.norm(regressed_velocity)

        ax.quiver(
            pos_x,
            pos_y,
            pos_z,
            vel_x,
            vel_y,
            vel_z,
            length=arrow_length * v_mag,
            color="red",
            label=f"Velocity before",
        )

        idx_after = key + idx_delta_after
        pos_x, pos_y, pos_z = ball_positions[idx_after]

        vel_x, vel_y, vel_z = ball_velocities[idx_after]
        v_mag = np.linalg.norm(ball_velocities[idx_after])

        if regression:
            ts = ball_time_stamps[idx_after : idx_after + num_reg_samples]
            ps = ball_positions[idx_after : idx_after + num_reg_samples]

            regressed_velocity = compute_velocity_regression(ts, ps, poly_deg)
            vel_x, vel_y, vel_z = regressed_velocity[0]
            v_mag = np.linalg.norm(regressed_velocity)

        ax.quiver(
            pos_x,
            pos_y,
            pos_z,
            vel_x,
            vel_y,
            vel_z,
            length=arrow_length * v_mag,
            color="green",
            label=f"Velocity after",
        )

        if value == ContactType.RACKET:
            idx_collision = key + idx_delta_racket
            pos_x, pos_y, pos_z = ball_positions[idx_collision]

            joint_angles = robot_joint_angles[idx_collision]

            normal = compute_racket_orientation(joint_angles)
            n_x, n_y, n_z = normal

            print(normal)

            ax.quiver(
                pos_x,
                pos_y,
                pos_z,
                n_x,
                n_y,
                n_z,
                length=10 * arrow_length * v_mag,
                color="orange",
                label=f"Racket normal",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Set the aspect ratio of the 3D plot to be equal
    ax.set_aspect("equal")


if __name__ == "__main__":
    visualize_regressed_velocity_vectors()

    plt.show()
