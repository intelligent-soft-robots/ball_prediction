import warnings
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union

from numpy import (
    abs,
    arctan2,
    array,
    cross,
    empty,
    empty_like,
    hstack,
    meshgrid,
    ndarray,
    pad,
    pi,
    polynomial,
    sqrt,
    zeros,
)
from numpy.linalg import norm


class ContactType(Enum):
    RACKET = "racket"
    TABLE = "table"
    UNKNOWN = "unknown"


WINDOW_SIZE = 10
POLYNOMIAL_DEGREE = 1
DETECTION_THRESHOLD = 0.15
TABLE_PLANE_HEIGHT = 0.77
TABLE_DETECTION_THRESHOLD = -0.03
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
    return_regression: bool = False,
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

    if return_regression:
        info = {"polynomial": polynomials}
        return regressed_state, info

    return regressed_state


def detect_rebounds(
    time_stamps: ndarray,
    positions: ndarray,
    velocities: Optional[ndarray] = None,
    window_size: int = WINDOW_SIZE,
    polynomial_degree: int = POLYNOMIAL_DEGREE,
    detection_threshold: float = DETECTION_THRESHOLD,
    table_height: float = TABLE_PLANE_HEIGHT,
    simulation_sample_delay: int = SIMULATION_DELAY,
    predictive_table_contact_detection: bool = False,
    return_states: bool = False,
) -> Union[List[int], Dict[int, ContactType]]:
    contact_dict = {}
    ball_state_history = []
    simulated_ball_state_history = []
    distances = []

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
        # ball_state: x, y, z, vx, vy, vz

        dt = time_stamps[i + simulation_sample_delay] - time_stamps[i]

        simulated_ball_state = step_ball_simulation(ball_state, dt)
        simulated_ball_state_history.append(simulated_ball_state)

        # Calculate the Euclidean distance between the simulated trajectory and actual positions
        if predictive_table_contact_detection:
            distance = norm(
                simulated_ball_state[:3] - positions[i + simulation_sample_delay, :3]
            )
        else:
            distance = norm(
                simulated_ball_state[:2] - positions[i + simulation_sample_delay, :2]
            )

        distances.append(distance)

        # Determine the contact type based on the rebound distance and table height
        contact = False

        if distance > detection_threshold:
            contact_type = ContactType.RACKET
            contact = True

        if distance > detection_threshold and simulated_ball_state[2] < table_height:
            contact_type = ContactType.TABLE
            contact = True

        if contact is True:
            # Add the rebound index and contact type to the respective lists
            contact_dict[i] = contact_type

    if return_states:
        info = {}

        info["ball_state_history"] = ball_state_history
        info["simulated_ball_state_history"] = simulated_ball_state_history
        info["distances"] = distances

        return contact_dict, info

    return contact_dict


def linear_table_contact(velocity_before_bounce):
    # assuming there is no friction effect, point contact

    velocity_after_bounce = velocity_before_bounce
    velocity_after_bounce = -velocity_after_bounce[2]
    return velocity_after_bounce


def lineare_racket_contact(
    velocity_before_bounce: Sequence[float], racket_orientation: Sequence[float]
):
    # assume no friction, point contact, no restitution, ball hits racket on flat surface,
    # surface is even
    velocity_after_bounce = velocity_before_bounce

    return velocity_after_bounce


def compute_racket_orientation(joint_angles_rad):
    quaternion = empty(4)
    normal_vector = empty(3)

    return quaternion, normal_vector


def calculate_rebound_velocity(V_initial, R_orient, C_normal):
    # Normalize vectors
    V_initial = array(V_initial) / norm(V_initial)
    C_normal = array(C_normal) / norm(C_normal)

    # Convert racket orientation to a unit quaternion
    R_quaternion = quat.from_float_array(R_orient)
    R_quaternion = quat.as_quat_array(R_quaternion / norm(R_quaternion))

    # Calculate reflection vector
    R_reflection = V_initial - 2 * np.dot(V_initial, C_normal) * C_normal

    # Transform reflection vector using quaternion rotation
    R_global = quat.as_float_array(
        R_quaternion * quat.as_quat_array(R_reflection) * quat.conjugate(R_quaternion)
    )

    # Return the rebound velocity vector
    return R_global.tolist()


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
                warnings.warn(
                    f"Combination values below threshold: {indices[i]}, {indices[j]}"
                )
                return True

    # If no combination satisfies the condition, return False
    return False


def velocity_regression(
    time_stamps: ndarray,
    positions: ndarray,
    polynomial_degree: int = 3,
) -> ndarray:
    velocities = empty_like(positions)

    position_polynomial = polynomial.Polynomial.fit(
        time_stamps, positions, deg=polynomial_degree
    )

    velocity_polynomial = position_polynomial.deriv()

    for axis in range(3):
        velocities[:, axis] = velocity_polynomial(time_stamps)

    return velocities


def estimate(
    time_stamps: Sequence[float],
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    racket_orientation: Optional[Sequence[float]] = None,
    regression: bool = True,
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
    velocities = array(velocities, copy=True)

    # get all rebounds from trajectory and specify if table or racket
    contact_dict = detect_rebounds(positions)
    rebound_indices = list(contact_dict.keys())

    check_difference_below_threshold(rebound_indices, n_regression_samples)

    # take velocity vector before bounce and after bounce
    for index, contact_type in contact_dict.items():
        if regression:
            time_stamps_before_bounce = time_stamps[
                index - n_regression_samples : index
            ]
            positions_before_bounce = positions[index - n_regression_samples : index]

            vel_before_bounce = velocity_regression(
                time_stamps_before_bounce, positions_before_bounce
            )
            vel_before_bounce = vel_before_bounce[-1]

            time_stamps_after_bounce = time_stamps[
                index + 1 : index + n_regression_samples
            ]
            positions_after_bounce = positions[index + 1 : index + n_regression_samples]

            vel_after_bounce = velocity_regression(
                time_stamps_after_bounce, positions_after_bounce
            )
            vel_after_bounce = vel_after_bounce[0]

        else:
            vel_before_bounce = velocities[index - 1]
            vel_after_bounce = velocities[index + 1]

        if ContactType.TABLE == contact_type:
            vel_after_bounce_no_spin = linear_table_contact(vel_before_bounce)

        if ContactType.RACKET == contact_type:
            vel_after_bounce_no_spin = lineare_racket_contact(
                vel_before_bounce, racket_orientation
            )

        vel_diff = vel_after_bounce - vel_after_bounce_no_spin

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

                return azimuth, elevation

        return vel_diff
