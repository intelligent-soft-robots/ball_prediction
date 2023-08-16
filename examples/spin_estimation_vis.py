import h5py
import matplotlib.pyplot as plt
from numpy import arange, array, convolve, linspace, ones, sort
from numpy.linalg import norm
from numpy.random import randint
from scipy.signal import savgol_filter

from ball_prediction.ball_prediction.contact_models.spin_estimator import (
    DETECTION_THRESHOLD,
    SIMULATION_DELAY,
    WINDOW_SIZE,
    ContactType,
    detect_rebounds,
    get_regressed_state,
    step_ball_simulation,
    filter_rebounds,
)

FILE_PATH = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/examples/data/ball_trajectories_with_return.hdf5"
INDEX = 59  # clean sample at 59


def test_ball_simulation():
    init_state = [
        0.87261184,
        -0.5063704,
        -0.43854895,
        -0.01799381,
        -4.56529991,
        4.76305174,
        0.0,
        0.0,
        0.0,
    ]
    n_time_steps = 100
    dt = 0.01

    time_stamps = linspace(0, dt * n_time_steps, n_time_steps)
    state_history = []

    ball_state = init_state
    for i in range(n_time_steps):
        ball_state = step_ball_simulation(ball_state, dt)
        state_history.append(ball_state)

    state_history = array(state_history)

    fig, axs = plt.subplots(9)
    for i in range(9):
        axs[i].scatter(time_stamps, state_history[:, i])


def visualize_data():
    marker_size = 1.25
    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    fig, axs = plt.subplots(6)
    for i in range(3):
        axs[i].scatter(ball_time_stamps, ball_positions[:, i], s=marker_size)
        axs[i + 3].scatter(ball_time_stamps, ball_velocities[:, i], s=marker_size)

    fig, axs = plt.subplots(3)
    axs[0].scatter(ball_positions[:, 0], ball_positions[:, 1], s=marker_size)
    axs[1].scatter(ball_positions[:, 1], ball_positions[:, 2], s=marker_size)
    axs[2].scatter(ball_positions[:, 0], ball_positions[:, 2], s=marker_size)


def visualize_data_3d():
    marker_size = 1.25
    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2])
    ax.set_aspect("equal")


def velocity_regression_visualisation():
    marker_size = 1.25
    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    ball_time_stamps = array(ball_time_stamps)
    ball_positions = array(ball_positions)

    start_regression = 10
    regression_window = WINDOW_SIZE

    regressed_state, info = get_regressed_state(
        time_stamps=ball_time_stamps[
            start_regression : start_regression + regression_window
        ],
        positions=ball_positions[
            start_regression : start_regression + regression_window, :
        ],
        polynomial_degree=3,
        return_info=True,
    )

    print(
        f"End state regression: {regressed_state[3:6]},"
        f"measured state: {ball_velocities[start_regression + regression_window]}"
    )

    fig, axs = plt.subplots(3)
    for axis in range(3):
        polynomial = info["polynomial"][axis]

        axs[axis].scatter(
            ball_time_stamps, ball_positions[:, axis], s=marker_size, c="#0193d7"
        )
        axs[axis].plot(*polynomial.linspace(), alpha=0.8, c="#CF5369")


def rebound_visualisation():
    marker_size = 1.75
    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    # clipping = 1

    # ball_time_stamps = ball_time_stamps[clipping:-clipping]
    # ball_positions = ball_positions[clipping:-clipping]
    # ball_velocities = ball_velocities[clipping:-clipping]

    # smoothing_window = 2

    # for axis in range(3):
    #    ball_positions[:, axis] = savgol_filter(
    #        ball_positions[:, axis], smoothing_window, 1
    #    )

    contact_dict, info = detect_rebounds(
        time_stamps=ball_time_stamps, positions=ball_positions, return_states=True
    )

    print(contact_dict)

    regressed_states = info["ball_state_history"]
    simulated_states = info["simulated_ball_state_history"]
    distances = info["xy_pred_errors"]
    distances_table = info["z_pred_errors"]

    regressed_states = array(regressed_states)
    simulated_states = array(simulated_states)

    # Plot regressed states
    fig, axs = plt.subplots(7, sharex=True)
    for i in range(3):
        axs[i].scatter(
            ball_time_stamps, ball_positions[:, i], s=marker_size, label="data"
        )

        axs[i + 3].scatter(
            ball_time_stamps, ball_velocities[:, i], s=marker_size, label="data"
        )

        axs[i].scatter(
            ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
            regressed_states[:, i],
            s=marker_size,
            label="regression",
        )

        axs[i + 3].scatter(
            ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
            regressed_states[:, i + 3],
            s=marker_size,
            label="regression",
        )

        axs[i].scatter(
            ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
            simulated_states[:, i],
            s=marker_size,
            label="simulated",
        )

        axs[i + 3].scatter(
            ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
            simulated_states[:, i + 3],
            s=marker_size,
            label="simulated",
        )

    for ax in axs:
        for index, contact_type in contact_dict.items():
            if contact_type == ContactType.RACKET:
                racket_color = "#17c7d0"
                ax.axvline(
                    ball_time_stamps[index],
                    color=racket_color,
                    linestyle="-",
                    alpha=0.5,
                    label="Racket Contact",
                )

            if contact_type == ContactType.TABLE:
                table_color = "#46b361"
                ax.axvline(
                    ball_time_stamps[index],
                    color=table_color,
                    linestyle="-",
                    alpha=0.5,
                    label="Table Contact",
                )

        axs[i].legend()
        axs[i + 3].legend()

    # Distance distribution
    axs[-1].scatter(
        ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
        distances,
        alpha=1.0,
        c="#0193d7",
        label="pred. error xy",
        s=marker_size,
    )
    axs[-1].scatter(
        ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
        distances_table,
        alpha=0.5,
        c="#CF5369",
        label="pred. error z",
        s=marker_size,
    )

    axs[-1].axhline(DETECTION_THRESHOLD, color="r", alpha=0.5)
    axs[-1].text(
        ball_time_stamps[-SIMULATION_DELAY],
        DETECTION_THRESHOLD,
        "Detection Threshold",
        horizontalalignment="right",
        verticalalignment="bottom",
        color="r",
        fontsize=8,
    )

    axs[-1].legend()


def compare_poly_degrees_visualisation():
    marker_size = 1.25
    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_positions = array(ball_positions)

    degrees = [1, 2, 3, 4, 5]
    distances_windows_table = []
    distances_windows_racket = []

    for deg in degrees:
        contact_dict, info = detect_rebounds(
            time_stamps=ball_time_stamps,
            positions=ball_positions,
            polynomial_degree=deg,
            predictive_table_contact_detection=True,
            return_states=True,
        )

        distances_windows_table.append(info["z_pred_errors"])
        distances_windows_racket.append(info["xy_pred_errors"])

    # Plot regressed states
    colors = [
        "#CF5369",
        "#0193d7",
        "#ffba4d",
        "#006c66",
        "#46b361",
        "#17c7d0",
        "#777777",
    ]

    n_plots = len(degrees) + 3
    fig, axs = plt.subplots(n_plots, sharex=True, constrained_layout=True)

    for i in range(3):
        axs[i].scatter(
            ball_time_stamps, ball_positions[:, i], s=marker_size, label="data"
        )

    axs[3].set_title("Distances of predicted to actual position")

    for i, distances in enumerate(distances_windows_table):
        axs[i + 3].scatter(
            ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
            distances,
            c=colors[i],
            alpha=0.9,
            marker=".",
            label="Table",
        )

    for i, distances in enumerate(distances_windows_racket):
        axs[i + 3].scatter(
            ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
            distances,
            c=colors[i],
            marker="x",
            alpha=0.6,
            label="Racket",
        )

    y_labels = ["x", "y", "z"]
    y_labels.extend([f"{i}. Order\n Polynomial" for i in degrees])

    for i in range(3):
        axs[i].set_ylabel(y_labels[i])

    for i in range(n_plots - 3):
        ax = axs[i + 3]
        ax.set_ylabel(r"$\Delta d_{pred}$")
        ax_right = ax.twinx()
        ax_right.set_ylabel(
            y_labels[i + 3],
            rotation="horizontal",
            labelpad=35,
            va="center",
            ha="center",
        )

    axs[-1].set_xlabel("Time t [s]")
    fig.align_ylabels()


def compare_windows_visualisation():
    marker_size = 1.25
    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]

    ball_time_stamps = array(ball_time_stamps)
    ball_positions = array(ball_positions)

    windows = [3, 5, 10, 20, 30, 50]
    distances_windows_table = []
    distances_windows_racket = []

    for window in windows:
        contact_dict, info = detect_rebounds(
            time_stamps=ball_time_stamps,
            positions=ball_positions,
            window_size=window,
            predictive_table_contact_detection=True,
            return_states=True,
        )

        distances_windows_table.append(info["z_pred_errors"])
        distances_windows_racket.append(info["xy_pred_errors"])

    # Plot regressed states
    colors = [
        "#CF5369",
        "#0193d7",
        "#ffba4d",
        "#006c66",
        "#46b361",
        "#17c7d0",
        "#777777",
    ]

    n_plots = len(windows) + 3
    fig, axs = plt.subplots(n_plots, sharex=True, constrained_layout=True)

    for i in range(3):
        axs[i].scatter(
            ball_time_stamps, ball_positions[:, i], s=marker_size, label="data"
        )

    axs[3].set_title("Distances of predicted to actual position")

    for i, distances in enumerate(distances_windows_table):
        axs[i + 3].scatter(
            ball_time_stamps[windows[i] : -SIMULATION_DELAY],
            distances,
            c=colors[i],
            alpha=0.9,
            marker=".",
            label="Table",
        )

    for i, distances in enumerate(distances_windows_racket):
        axs[i + 3].scatter(
            ball_time_stamps[windows[i] : -SIMULATION_DELAY],
            distances,
            c=colors[i],
            marker="x",
            alpha=0.6,
            label="Racket",
        )

    y_labels = ["x", "y", "z"]
    y_labels.extend([f"{i} Window" for i in windows])

    for i in range(3):
        axs[i].set_ylabel(y_labels[i])

    for i in range(n_plots - 3):
        ax = axs[i + 3]
        ax.set_ylabel(r"$\Delta d_{pred}$")
        ax_right = ax.twinx()
        ax_right.set_ylabel(
            y_labels[i + 3],
            rotation="horizontal",
            labelpad=35,
            va="center",
            ha="center",
        )

    axs[-1].set_xlabel("Time t [s]")
    fig.align_ylabels()


def load_data():
    file = h5py.File(FILE_PATH, "r")
    collection = {}
    for index in list(file["originals"].keys()):
        collection[index] = {
            "launch_parameter": array(file["originals"][index]["launch_parameter"]),
            "date_stamp": file["originals"][index]["date_stamp"],
            "time_stamps": array(file["originals"][index]["time_stamps"]),
            "ball_ids": array(file["originals"][index]["ball_ids"]),
            "ball_time_stamps": array(file["originals"][index]["ball_time_stamps"]),
            "ball_positions": array(file["originals"][index]["ball_positions"]),
            "ball_velocities": array(file["originals"][index]["ball_velocities"]),
            "robot_time_stamps": array(file["originals"][index]["robot_time_stamps"]),
            "robot_joint_angles": array(file["originals"][index]["robot_joint_angles"]),
            "robot_joint_angles_desired": array(
                file["originals"][index]["robot_joint_angles_desired"]
            ),
            "robot_joint_angle_velocities": array(
                file["originals"][index]["robot_joint_angle_velocities"]
            ),
            "robot_joint_angle_velocities_desired": array(
                file["originals"][index]["robot_joint_angle_velocities_desired"]
            ),
            "robot_pressures_agonist": array(
                file["originals"][index]["robot_pressures_agonist"]
            ),
            "robot_pressures_antagonist": array(
                file["originals"][index]["robot_pressures_antagonist"]
            ),
        }

    return collection


def test_indicies_compare():
    from ball_prediction.ball_prediction.contact_models.spin_estimator import check_difference_below_threshold

    threshold = 5
    n_runs = 100

    for i in range(n_runs):
        rebounds_indices = randint(0, 200, (5,))

        rebounds_indices = sort(rebounds_indices)

        if check_difference_below_threshold(rebounds_indices, threshold=threshold):
            print(f"Difference to small: {rebounds_indices}")


def test_spin_estimator():
    from ball_prediction.ball_prediction.contact_models.spin_estimator import estimate

    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]

    ball_time_stamps = array(ball_time_stamps)
    ball_positions = array(ball_positions)

    output = estimate(ball_time_stamps, ball_positions)

    print(output)


def test_contact_dict_filter():
    marker_size = 1.75
    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    contact_dict, info = detect_rebounds(
        time_stamps=ball_time_stamps, positions=ball_positions, return_states=True
    )

    filtered_contact_dict = filter_rebounds(
        contact_dict, ball_time_stamps, time_threshold=0.05, time_filter_stamp_range=5
    )

    print(contact_dict)
    print(filtered_contact_dict)

    regressed_states = info["ball_state_history"]
    simulated_states = info["simulated_ball_state_history"]
    distances = info["xy_pred_errors"]
    distances_table = info["z_pred_errors"]

    regressed_states = array(regressed_states)
    simulated_states = array(simulated_states)

    # Plot regressed states
    fig, axs = plt.subplots(7, sharex=True)
    for i in range(3):
        axs[i].scatter(
            ball_time_stamps, ball_positions[:, i], s=marker_size, label="data"
        )

        axs[i + 3].scatter(
            ball_time_stamps, ball_positions[:, i], s=marker_size, label="data"
        )

    for i in range(3):
        for index, contact_type in contact_dict.items():
            if contact_type == ContactType.RACKET:
                racket_color = "#17c7d0"
                axs[i].axvline(
                    ball_time_stamps[index],
                    color=racket_color,
                    linestyle="-",
                    alpha=0.5,
                    label="Racket Contact",
                )

            if contact_type == ContactType.TABLE:
                table_color = "#46b361"
                axs[i].axvline(
                    ball_time_stamps[index],
                    color=table_color,
                    linestyle="-",
                    alpha=0.5,
                    label="Table Contact",
                )

    for i in range(3, 6):
        for index, contact_type in filtered_contact_dict.items():
            if contact_type == ContactType.RACKET:
                racket_color = "#17c7d0"
                axs[i].axvline(
                    ball_time_stamps[index],
                    color=racket_color,
                    linestyle="-",
                    alpha=0.5,
                    label="Racket Contact",
                )

            if contact_type == ContactType.TABLE:
                table_color = "#46b361"
                axs[i].axvline(
                    ball_time_stamps[index],
                    color=table_color,
                    linestyle="-",
                    alpha=0.5,
                    label="Table Contact",
                )

    # Distance distribution
    axs[-1].scatter(
        ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
        distances,
        alpha=1.0,
        c="#0193d7",
        label="pred. error xy",
        s=marker_size,
    )
    axs[-1].scatter(
        ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY],
        distances_table,
        alpha=0.5,
        c="#CF5369",
        label="pred. error z",
        s=marker_size,
    )


def visualize_regressed_velocity_vectors():
    marker_size = 1.25
    index = str(INDEX)

    arrow_length = 0.01
    idx_delta = 2

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

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
        idx_before = key - idx_delta
        pos_x, pos_y, pos_z = ball_positions[idx_before]
        vel_x, vel_y, vel_z = ball_velocities[idx_before]
        v_mag = norm(ball_velocities[idx_before])

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

        idx_after = key + idx_delta
        pos_x, pos_y, pos_z = ball_positions[idx_after]
        vel_x, vel_y, vel_z = ball_velocities[idx_after]
        v_mag = norm(ball_velocities[idx_after])

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

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Set the aspect ratio of the 3D plot to be equal
    ax.set_aspect("equal")
    pass


def test_orientation_calculation():
    import math

    import numpy as np
    from scipy.spatial.transform import Rotation

    from ball_prediction.ball_prediction.contact_models.spin_estimator import compute_racket_orientation

    np.set_printoptions(suppress=True)
    pi = math.pi

    joint_angles_rad = [40, 00, 0.0, 0.0]
    length_upper_arm = 0.5
    lenght_lower_arm = 0.4
    degrees = True

    theta_1, theta_2, theta_3, theta_4 = joint_angles_rad

    o = [0.0, 0.0, 0.0]

    o_0 = [0.0, 1.0, 0.0]

    R1 = Rotation.from_rotvec(np.deg2rad([0.0, 0.0, theta_1]))  # Shoulder joint
    R2 = Rotation.from_rotvec(np.deg2rad([0.0, theta_2, 0.0]))  # Shoulder joint

    R12 = R2 * R1
    T1 = [0, 0, length_upper_arm]
    R3 = Rotation.from_rotvec(np.deg2rad([0.0, theta_3, 0.0]))  # Elbow joint
    T2 = [0, 0, lenght_lower_arm]
    R4 = Rotation.from_rotvec(np.deg2rad([0.0, 0.0, theta_4]))  # Wrist joint

    o_2 = R12.apply(o_0)
    o_2 += T1

    o_3 = R3.apply(o_2)
    o_3 += T2

    o_4 = R4.apply(o_3)

    # Visualisation
    ax = plt.figure().add_subplot(projection="3d")
    x_axis = np.array([0.4, 0, 0])
    y_axis = np.array([0, 0.4, 0])
    z_axis = np.array([0, 0, 0.4])

    # Draw the basis vectors as arrows
    ax.quiver(*o, *x_axis, color="r", label="X-axis")
    ax.quiver(*o, *y_axis, color="g", label="Y-axis")
    ax.quiver(*o, *z_axis, color="b", label="Z-axis")

    ax.scatter(*o, s=20, c="r", alpha=0.5)

    # Shoulder
    P0 = o
    print(f"P0: {P0}")
    ax.scatter(*P0, s=20, c="b", alpha=0.5)

    # Elbow
    P1 = o_2
    print(f"P1: {P1}")
    ax.scatter(*P1, s=20, c="g", alpha=0.5)

    # Wrist
    P2 = o_3
    print(f"P2: {P2}")
    ax.scatter(*P2, s=20, c="black", alpha=0.5)

    # Shoulder to Elbow
    P0P1 = [[i, j] for i, j in zip(P0, P1)]
    print(P0P1)
    ax.plot(*P0P1)

    # Elbow to Wrist
    P1P2 = [[i, j] for i, j in zip(P1, P2)]
    print(P1P2)
    ax.plot(*P1P2)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pass


def test_orientation_calculation_homogenous():
    import math

    import numpy as np
    from scipy.spatial.transform import Rotation

    from ball_prediction.ball_prediction.contact_models.spin_estimator import compute_racket_orientation

    np.set_printoptions(suppress=True)
    pi = math.pi

    joint_angles_deg = [20, 30, 00, 0]
    joint_angles_rad = np.deg2rad(joint_angles_deg)
    length_upper_arm = 0.5
    length_lower_arm = 0.4

    theta_1, theta_2, theta_3, theta_4 = joint_angles_rad

    o = [0.0, 0.0, 0.0]

    def translation_matrix(x, y, z):
        return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

    def rotation_matrix_x(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    def rotation_matrix_y(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    def rotation_matrix_z(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    print(theta_1)
    print(theta_2)
    print(theta_3)
    print(theta_4)

    T_10 = rotation_matrix_z(theta_1)
    T_21 = rotation_matrix_y(theta_2)
    T_32 = translation_matrix(0, 0, length_upper_arm)

    T_43 = rotation_matrix_y(theta_3)
    T_54 = translation_matrix(0, 0, length_lower_arm)
    T_65 = rotation_matrix_z(theta_4)

    origin = np.array([0, 0, 0])

    homogeneous_coordinate = np.append(origin, 1)

    T_20 = T_21 @ T_10
    T_30 = T_32 @ T_21 @ T_10
    T_40 = T_43 @ T_32 @ T_21 @ T_10
    T_50 = T_54 @ T_43 @ T_32 @ T_21 @ T_10
    T_60 = T_65 @ T_54 @ T_43 @ T_32 @ T_21 @ T_10

    end_point_00 = homogeneous_coordinate  # origin
    end_point_10 = T_10 @ end_point_00
    end_point_20 = T_21 @ end_point_10
    end_point_30 = T_32 @ end_point_20  # joint 1
    end_point_40 = T_43 @ end_point_30
    end_point_50 = T_54 @ end_point_40  # joint 2
    end_point_60 = T_65 @ end_point_50

    # Plotting the 3D visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the joints
    joints = [end_point_00[:3], end_point_30[:3], end_point_50[:3]]
    for i, joint in enumerate(joints):
        ax.scatter(joint[0], joint[1], joint[2], c="r", marker="o", s=36)
        ax.text(
            joint[0],
            joint[1],
            joint[2],
            f"Joint {i}",
            fontsize=12,
            ha="right",
            va="bottom",
        )

    # Set axis labels (optional)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set the aspect ratio of the 3D plot to be equal
    ax.set_aspect("equal")

    # Add a legend for the coordinate system axes
    ax.legend()


def visualize_velocity_vectors():
    marker_size = 1.25
    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2])
    ax.set_aspect("equal")


if __name__ == "__main__":
    # visualize_data()
    # visualize_data_3d()
    # velocity_regression_visualisation()
    # rebound_visualisation()
    # test_ball_simulation()
    # compare_poly_degrees_visualisation()
    # compare_windows_visualisation()

    # test_orientation_calculation()
    # test_orientation_calculation_homogenous()

    # test_indicies_compare()
    # test_spin_estimator()

    # test_contact_dict_filter()

    visualize_regressed_velocity_vectors()

    plt.show()
