from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.signal import find_peaks

from ball_prediction.models.magnus_regressor import compute_velocity_regression
from ball_prediction.models.utils import ContactType

WINDOW_SIZE = 10
POLYNOMIAL_DEGREE = 1
DETECTION_THRESHOLD = 0.10
DETECTION_RANGE = 10
TABLE_PLANE_HEIGHT = 0.77
TABLE_DETECTION_THRESHOLD = 0.10
TABLE_DETECTION_SAMPLE_DISTANCE = 10
SIMULATION_DELAY = 5


def detect_rebounds(
    time_stamps: np.ndarray,
    positions: np.ndarray,
    velocities: Optional[np.ndarray] = None,
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
            ball_state = np.hstack((positions[i], velocities[i]))
        else:
            time_stamps_window = time_stamps[i - window_size : i]
            positions_window = positions[i - window_size : i, :]

            ball_state = get_regressed_state(
                time_stamps_window, positions_window, polynomial_degree
            )

        ball_state = np.pad(ball_state, (0, 3), mode="constant")
        ball_state_history.append(ball_state)

        # Simulate the trajectory using the small model
        dt = time_stamps[i + simulation_sample_delay] - time_stamps[i]

        simulated_ball_state = step_ball_simulation(ball_state, dt)
        simulated_ball_state_history.append(simulated_ball_state)

        # Calculate the Euclidean distance between the simulated trajectory and actual positions
        xy_error = np.linalg.norm(
            simulated_ball_state[:2] - positions[i + simulation_sample_delay, :2]
        )
        xy_pred_errors.append(xy_error)

        z_error = np.linalg.norm(
            simulated_ball_state[2] - positions[i + simulation_sample_delay, 2]
        )
        z_pred_errors.append(z_error)

        total_error = np.linalg.norm(
            simulated_ball_state[:3] - positions[i + simulation_sample_delay, :3]
        )
        total_pred_errors.append(total_error)

    xy_pred_errors = np.array(xy_pred_errors)
    z_pred_errors = np.array(z_pred_errors)
    total_pred_errors = np.array(total_pred_errors)

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


def step_ball_simulation(ball_state, dt):
    # Constants
    rho = 1.18  # Air density
    c_drag = 0.47016899  # Drag coefficient
    c_lift = 1.46968343  # Magnus coefficient
    g = 9.80801  # Acceleration due to gravity
    r = 0.02  # Radius of the ball
    m = 0.0027  # Mass of the ball
    A = np.pi * r**2  # Cross-sectional area of the ball

    k_magnus = 0.5 * rho * c_lift * A * r / m
    k_drag = -0.5 * rho * c_drag * A / m
    k_gravity = g

    # Step calculation
    q = np.array(ball_state)
    v = q[3:6]
    omega = q[6:9]

    F_gravity = k_gravity * np.array([0, 0, -1])
    F_drag = k_drag * np.linalg.norm(v) * v
    F_magnus = k_magnus * np.cross(omega, v)

    # System dynamics
    dv_dt = F_gravity + F_drag + F_magnus

    domega_dt = np.zeros(3)
    dq_dt = np.hstack((v, dv_dt, domega_dt))

    q_next = q + dt * dq_dt

    return q_next


def get_regressed_state(
    time_stamps: np.ndarray,
    positions: np.ndarray,
    polynomial_degree: int,
    return_info: bool = False,
):
    regressed_state = np.empty(6)
    polynomials = []

    for axis in range(3):
        position_polynomial = np.polynomial.Polynomial.fit(
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
