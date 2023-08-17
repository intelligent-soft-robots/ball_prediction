import logging
import warnings
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union

from numpy import (
    abs,
    arctan2,
    array,
    cos,
    cross,
    empty,
    empty_like,
    hstack,
    meshgrid,
    ndarray,
    pad,
    pi,
    polynomial,
    sin,
    sqrt,
    zeros,
)
from numpy.linalg import norm
from scipy.signal import find_peaks


class ContactType(Enum):
    RACKET = "racket"
    TABLE = "table"
    UNKNOWN = "unknown"


WINDOW_SIZE = 10
POLYNOMIAL_DEGREE = 1
DETECTION_THRESHOLD = 0.10
DETECTION_RANGE = 10
TABLE_PLANE_HEIGHT = 0.77
TABLE_DETECTION_THRESHOLD = 0.10
TABLE_DETECTION_SAMPLE_DISTANCE = 10
SIMULATION_DELAY = 5


def step_ball_simulation(ball_state, dt):
    # Constants
    rho = 1.18  # Air density
    c_drag = 0.47016899  # Drag coefficient
    c_lift = 1.46968343  # Magnus coefficient
    g = 9.80801  # Acceleration due to gravity
    r = 0.02  # Radius of the ball
    m = 0.0027  # Mass of the ball
    A = pi * r**2  # Cross-sectional area of the ball

    k_magnus = 0.5 * rho * c_lift * A * r / m
    k_drag = -0.5 * rho * c_drag * A / m
    k_gravity = g

    # Step calculation
    q = array(ball_state)
    v = q[3:6]
    omega = q[6:9]

    F_gravity = k_gravity * array([0, 0, -1])
    F_drag = k_drag * norm(v) * v
    F_magnus = k_magnus * cross(omega, v)

    # System dynamics
    dv_dt = F_gravity + F_drag + F_magnus

    domega_dt = zeros(3)
    dq_dt = hstack((v, dv_dt, domega_dt))

    q_next = q + dt * dq_dt

    return q_next


def get_regressed_state(
    time_stamps: ndarray,
    positions: ndarray,
    polynomial_degree: int,
    return_info: bool = False,
):
    regressed_state = empty(6)
    polynomials = []

    for axis in range(3):
        position_polynomial = polynomial.Polynomial.fit(
            time_stamps, positions[:, axis], deg=polynomial_degree
        )
        polynomials.append(position_polynomial)

        regressed_state[axis] = position_polynomial(
            time_stamps[-1]
        )  # Store last position
        regressed_state[axis + 3] = position_polynomial.deriv()(
            time_stamps[-1]
        )  # Store last velocity

    if return_info:
        info = {"polynomial": polynomials}
        return regressed_state, info

    return regressed_state


def velocity_regression(
    time_stamps: ndarray,
    positions: ndarray,
    polynomial_degree: int = POLYNOMIAL_DEGREE,
) -> ndarray:
    velocities = empty_like(positions)

    for axis in range(3):
        position_polynomial = polynomial.Polynomial.fit(
            time_stamps, positions[:, axis], deg=polynomial_degree
        )

        velocity_polynomial = position_polynomial.deriv()

        velocities[:, axis] = velocity_polynomial(time_stamps)

    return velocities


def detect_rebounds(
    time_stamps: ndarray,
    positions: ndarray,
    velocities: Optional[ndarray] = None,
    window_size: int = WINDOW_SIZE,
    polynomial_degree: int = POLYNOMIAL_DEGREE,
    detection_threshold: float = DETECTION_THRESHOLD,
    detection_range: int = DETECTION_RANGE,
    table_height: float = TABLE_PLANE_HEIGHT,
    detection_threshold_table: float = TABLE_DETECTION_THRESHOLD,
    simulation_sample_delay: int = SIMULATION_DELAY,
    predictive_table_contact_detection: bool = False,
    return_states: bool = False,
) -> Union[List[int], Dict[int, ContactType]]:
    contact_dict = {}
    ball_state_history = []
    simulated_ball_state_history = []

    z_pred_errors = []
    xy_pred_errors = []
    total_pred_errors = []

    for i in range(window_size, len(positions) - simulation_sample_delay):
        if velocities is not None:
            ball_state = hstack((positions[i], velocities[i]))
        else:
            time_stamps_window = time_stamps[i - window_size : i]
            positions_window = positions[i - window_size : i, :]

            ball_state = get_regressed_state(
                time_stamps_window, positions_window, polynomial_degree
            )

        ball_state = pad(ball_state, (0, 3), mode="constant")
        ball_state_history.append(ball_state)

        # Simulate the trajectory using the small model
        dt = time_stamps[i + simulation_sample_delay] - time_stamps[i]

        simulated_ball_state = step_ball_simulation(ball_state, dt)
        simulated_ball_state_history.append(simulated_ball_state)

        # Calculate the Euclidean distance between the simulated trajectory and actual positions
        xy_error = norm(
            simulated_ball_state[:2] - positions[i + simulation_sample_delay, :2]
        )
        xy_pred_errors.append(xy_error)

        z_error = norm(
            simulated_ball_state[2] - positions[i + simulation_sample_delay, 2]
        )
        z_pred_errors.append(z_error)

        total_error = norm(
            simulated_ball_state[:3] - positions[i + simulation_sample_delay, :3]
        )
        total_pred_errors.append(total_error)

    xy_pred_errors = array(xy_pred_errors)
    z_pred_errors = array(z_pred_errors)
    total_pred_errors = array(total_pred_errors)

    # Find peaks in the predicted errors between data and simulation
    contact_indices_racket = find_peaks(
        xy_pred_errors, height=detection_threshold, distance=detection_range
    )[0]
    contact_indices_racket = [i + window_size for i in contact_indices_racket]

    if predictive_table_contact_detection:
        positions_inv = positions - table_height
        positions_inv *= -1

        contact_indices_table = find_peaks(
            positions_inv[:, 2],
            height=detection_threshold_table,
            distance=detection_range,
        )[0]
    else:
        contact_indices_table = find_peaks(
            z_pred_errors, height=detection_threshold, distance=detection_range
        )[0]
        contact_indices_table = [i + window_size for i in contact_indices_table]

    # Assign contact indices to contact dict
    for index in contact_indices_racket:
        contact_type = ContactType.RACKET
        contact_dict[index] = contact_type

    for index in contact_indices_table:
        contact_type = ContactType.TABLE
        contact_dict[index] = contact_type

    if return_states:
        info = {}

        info["ball_state_history"] = ball_state_history
        info["simulated_ball_state_history"] = simulated_ball_state_history
        info["xy_pred_errors"] = xy_pred_errors
        info["z_pred_errors"] = z_pred_errors
        info["total_pred_errors"] = total_pred_errors

        return contact_dict, info

    return contact_dict


def linear_table_contact(velocity_before_bounce):
    # assuming there is no friction effect, point contact

    velocity_after_bounce = velocity_before_bounce
    velocity_after_bounce = -velocity_after_bounce[2]
    return velocity_after_bounce


def lineare_racket_contact(
    velocity_before_bounce: Sequence[float], robot_joint_angles_rad: Sequence[float]
):
    racket_normal = compute_racket_orientation(joint_angles_rad=robot_joint_angles_rad)
    velocity_after_bounce = calculate_rebound_velocity(
        velocity_before_bounce, racket_normal
    )

    return velocity_after_bounce


def compute_racket_orientation(
    joint_angles_rad,
):
    theta_1, theta_2, theta_3, theta_4 = joint_angles_rad

    def rotation_matrix_x(theta):
        c = cos(theta)
        s = sin(theta)
        return array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    def rotation_matrix_y(theta):
        c = cos(theta)
        s = sin(theta)
        return array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    def rotation_matrix_z(theta):
        c = cos(theta)
        s = sin(theta)
        return array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Assuming robot is set up not perfectly within tennicam coordinates
    theta_x_error = 0.0
    theta_y_error = 0.0
    theta_z_error = 0.0

    T_X = rotation_matrix_x(theta_x_error)
    T_Y = rotation_matrix_y(theta_y_error)
    T_Z = rotation_matrix_z(theta_z_error)

    T_COMP = T_Z @ T_Y @ T_X

    racket_normal = [0.0, 1.0, 0.0, 1.0]

    T_10 = rotation_matrix_z(theta_1)
    T_21 = rotation_matrix_y(theta_2)
    T_43 = rotation_matrix_y(theta_3)
    T_65 = rotation_matrix_z(theta_4)

    T_60 = T_65 @ T_43 @ T_21 @ T_10

    trans_racket_normal = T_60 @ T_COMP @ racket_normal

    return trans_racket_normal


def filter_rebounds(
    contact_dict: dict,
    time_stamps: Optional[Sequence[float]] = None,
    index_threshold: int = 10,
    time_threshold: float = 0.1,
    time_filter_stamp_range=5,
):
    filtered_contact_dict = contact_dict.copy()

    print(f"Contact dict: {filtered_contact_dict}")

    # remove rebound from index if there is less then threshold measurement samples before and after
    keys_to_remove = []

    for key, value in filtered_contact_dict.items():
        print(key)
        # Samples below threshold are removed. At the border of the ball tracking system often noisy
        # samples can be found.
        if key < index_threshold:
            keys_to_remove.append(key)
            continue

        # Samples above threshold are removed. At the border of the ball tracking system often noisy
        # samples can be found.
        if key > len(time_stamps) - index_threshold:
            keys_to_remove.append(key)
            continue

        # Check if there is a sudden time jump which resulted in a false detected ball contact.
        # There should be no large gap and therefore the sample time stamps before and after should be
        # smaller the the given time threshold. We assume 200Hz sampling. So 5 samples should be
        # collected in 0.025 s.
        contact_ts = time_stamps[key]
        contact_ts_lower = time_stamps[key - time_filter_stamp_range]
        contact_ts_upper = time_stamps[key + time_filter_stamp_range]

        if contact_ts - contact_ts_lower > time_threshold:
            keys_to_remove.append(key)
            continue

        if contact_ts_upper - contact_ts > time_threshold:
            keys_to_remove.append(key)
            continue

    print(keys_to_remove)
    for key in keys_to_remove:
        del filtered_contact_dict[key]

    return filtered_contact_dict


def calculate_rebound_velocity(v_before, racket_normal):
    # ! Assumptions:
    # Point-contact of ball and surface, ball is not spinning,
    # ball does not lose energy due to friction or slip
    # ball and racket does not lose energy due to elastic deformation of the ball
    # all energy is fully maintained.
    # surface is perfectly flat and is fully described by normal vector.

    v_before = array(v_before)
    racket_normal = array(racket_normal)

    # Normalize vectors
    racket_normal = racket_normal / norm(racket_normal)

    # Calculate the component of v_in that is perpendicular to the surface
    v_perpendicular = (v_before @ racket_normal) * racket_normal

    # Calculate the component of v_in that is parallel to the surface
    v_parallel = v_before - v_perpendicular

    # Calculate the velocity vector of the ball after the collision (v_out)
    return v_parallel - v_perpendicular


def check_difference_below_threshold(indices, threshold):
    indices = array(indices)

    # Get the total number of elements
    num_elements = len(indices)

    # Generate all combinations of indices (2 at a time)
    indices = array(list(range(num_elements)))
    combinations = array(meshgrid(indices, indices)).T.reshape(-1, 2)

    # Iterate over each combination and check if the difference is below the threshold
    for combination in combinations:
        i, j = combination
        if i < j:
            diff = abs(indices[i] - indices[j])
            if diff <= threshold:
                logging.debug(
                    f"Combination values below threshold: {indices[i]}, {indices[j]}"
                )
                return True

    # If no combination satisfies the condition, return False
    return False


def estimate(
    time_stamps: Sequence[float],
    positions: Sequence[Sequence[float]],
    velocities: Optional[Sequence[Sequence[float]]] = None,
    racket_orientation: Optional[Sequence[float]] = None,
    n_regression_samples: int = 10,
    return_polar: bool = False,
):
    """Evaluates the effects of spin of a table tennis ball at rebound.

    This function neglects effects like friction, stiction. It assumes perfectly
    even table and racket surfaces and no energy loss due to elastic effects.

    Args:
        positions (Sequence[Sequence[float]]): _description_
        velocities (Sequence[Sequence[float]]): _description_
        regression (bool, optional): _description_. Defaults to True.
        use_invertable_contact_model (bool, optional): _description_. Defaults to False.
    """
    time_stamps = array(time_stamps, copy=True)
    positions = array(positions, copy=True)

    # get all rebounds from trajectory and specify if table or racket
    contact_dict = detect_rebounds(time_stamps, positions)
    contact_dict = filter_rebounds(contact_dict)

    rebound_indices = list(contact_dict.keys())

    if check_difference_below_threshold(rebound_indices, n_regression_samples):
        logging.warning("Identified rebound to close.")

    contact_list = []

    # take velocity vector before bounce and after bounce
    for index, contact_type in contact_dict.items():
        if velocities is None:
            # Regression before bounce
            time_stamps_before_bounce = time_stamps[
                index - n_regression_samples : index
            ]
            positions_before_bounce = positions[index - n_regression_samples : index]

            vel_before_bounce = velocity_regression(
                time_stamps_before_bounce, positions_before_bounce
            )
            vel_before_bounce = vel_before_bounce[-1]

            # Regression after bounce
            time_stamps_after_bounce = time_stamps[
                index + 1 : index + n_regression_samples
            ]
            positions_after_bounce = positions[index + 1 : index + n_regression_samples]

            vel_after_bounce = velocity_regression(
                time_stamps_after_bounce, positions_after_bounce
            )
            vel_after_bounce = vel_after_bounce[0]
        else:
            velocities = array(velocities, copy=True)

            vel_before_bounce = velocities[index - 1]
            vel_after_bounce = velocities[index + 1]

        if ContactType.TABLE == contact_type:
            vel_after_bounce_no_spin = linear_table_contact(vel_before_bounce)

        if ContactType.RACKET == contact_type:
            vel_after_bounce_no_spin = lineare_racket_contact(
                vel_before_bounce, racket_orientation
            )

        vel_diff = vel_after_bounce - vel_after_bounce_no_spin

        spin_effect = vel_diff

        if return_polar:
            v_x = vel_diff[0]
            v_y = vel_diff[1]
            v_z = vel_diff[2]

            if v_x == 0 and v_y == 0:
                azimuth = 0
            else:
                planar_magnitude = sqrt(v_x**2 + v_y**2)

                azimuth = arctan2(v_y, v_x)
                elevation = arctan2(v_z, planar_magnitude)

                spin_effect = [azimuth, elevation]

        contact_dict[index] = [contact_type, spin_effect]
        contact_list.append((index, contact_type, spin_effect))

    return contact_dict
