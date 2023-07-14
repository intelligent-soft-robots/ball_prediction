import h5py
import matplotlib.pyplot as plt
from numpy import array, ones, convolve
from scipy.signal import savgol_filter

from ball_prediction.spin_estimator import detect_rebounds, get_regressed_state

FILE_PATH = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/examples/data/ball_trajectories_with_return.hdf5"
INDEX = 28

WINDOW_SIZE = 5
SIMULATION_DELAY = 1


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
    marker_size = 1.25
    index = str(INDEX)

    collection = load_data()

    ball_time_stamps = collection[index]["ball_time_stamps"]
    ball_positions = collection[index]["ball_positions"]
    ball_velocities = collection[index]["ball_velocities"]

    clipping = 1

    ball_time_stamps = ball_time_stamps[clipping:-clipping]
    ball_positions = ball_positions[clipping:-clipping]
    ball_velocities = ball_velocities[clipping:-clipping]

    smoothing_window = 2

    for axis in range(3):
        ball_positions[:,axis] = savgol_filter(ball_positions[:,axis], smoothing_window, 1)

    print(len(ball_time_stamps))
    print(len(ball_positions))

    contact_dict, info = detect_rebounds(
        time_stamps=ball_time_stamps, positions=ball_positions, return_states=True
    )

    regressed_states = info["regressed_ball_states"]
    simulated_states = info["simulated_ball_states"]
    distances = info["distances"]

    regressed_states = array(regressed_states)
    simulated_states = array(simulated_states)

    print(len(regressed_states))

    # Plot regressed states
    fig, axs = plt.subplots(6)
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

        axs[i].legend()
        axs[i + 3].legend()

    # Distance distribution
    fig, axs = plt.subplots()
    axs.plot(ball_time_stamps[WINDOW_SIZE:-SIMULATION_DELAY], distances)


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

    plt.show()
