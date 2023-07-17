import h5py
import matplotlib.pyplot as plt
from numpy import arange, array, convolve, linspace, ones
from scipy.signal import savgol_filter

from ball_prediction.spin_estimator import (
    DETECTION_THRESHOLD,
    SIMULATION_DELAY,
    WINDOW_SIZE,
    ContactType,
    detect_rebounds,
    get_regressed_state,
    step_ball_simulation,
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

    plt.show()


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
        return_regression=True,
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


if __name__ == "__main__":
    # visualize_data()
    # velocity_regression_visualisation()
    rebound_visualisation()
    # test_ball_simulation()
    # compare_poly_degrees_visualisation()
    # compare_windows_visualisation()

    plt.show()
