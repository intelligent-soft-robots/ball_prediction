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
    meshgrid,
    ndarray,
    pi,
    polynomial,
    sqrt,
)
from numpy.linalg import norm


class ContactType(Enum):
    RACKET = "racket"
    TABLE = "table"
    UNKNOWN = "unknown"


POLYNOMIAL_DEGREE = 3
DETECTION_THRESHOLD = 0.1
TABLE_PLANE_HEIGHT = 0.77
TABLE_DETECTION_THRESHOLD = -0.03
TABLE_DETECTION_SAMPLE_DISTANCE = 10


def step_ball_simulation(ball_state, dt, integration_method: str = "euler_forward"):
    # Constants
    rho = 1.2  # Air density
    Cd = 0.5  # Drag coefficient
    g = 9.8  # Acceleration due to gravity
    r = 0.1  # Radius of the ball
    m = 0.5  # Mass of the ball
    A = pi * r**2  # Cross-sectional area of the ball
    S = pi * r**2  # Reference area for the Magnus effect

    # Unpack ball state
    x, y, z, vx, vy, vz, omegax, omegay, omegaz = ball_state

    # Calculate the air resistance and Magnus force
    v = sqrt(vx**2 + vy**2 + vz**2)
    F_drag = -0.5 * Cd * rho * A * v * array([vx, vy, vz])
    F_gravity = array([0, 0, -m * g])
    F_magnus = (
        0.5 * Cd * rho * S * cross(array([omegax, omegay, omegaz]), array([vx, vy, vz]))
    )

    # Calculate the acceleration and angular acceleration
    a = (F_drag + F_gravity + F_magnus) / m

    # Perform integration using the selected method
    if integration_method == "semi_euler_forward":
        ball_state += dt * array([vx, vy, vz, a[0], a[1], a[2], omegax, omegay, omegaz])
    elif integration_method == "rk4":
        k1 = dt * array([vx, vy, vz, a[0], a[1], a[2], omegax, omegay, omegaz])
        k2 = dt * array(
            [
                vx + 0.5 * k1[3],
                vy + 0.5 * k1[4],
                vz + 0.5 * k1[5],
                a[0],
                a[1],
                a[2],
                omegax,
                omegay,
                omegaz,
            ]
        )
        k3 = dt * array(
            [
                vx + 0.5 * k2[3],
                vy + 0.5 * k2[4],
                vz + 0.5 * k2[5],
                a[0],
                a[1],
                a[2],
                omegax,
                omegay,
                omegaz,
            ]
        )
        k4 = dt * array(
            [
                vx + k3[3],
                vy + k3[4],
                vz + k3[5],
                a[0],
                a[1],
                a[2],
                omegax,
                omegay,
                omegaz,
            ]
        )
        ball_state += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return ball_state


def get_regressed_state(
    time_stamps: ndarray,
    positions: ndarray,
    polynomial_degree: int,
):
    regressed_state = empty(6)

    for axis in range(3):
        position_polynomial = polynomial.Polynomial.fit(
            time_stamps, positions[:, axis], deg=polynomial_degree
        )

        regressed_state[axis] = position_polynomial(
            time_stamps[-1]
        )  # Store last position
        regressed_state[axis + 3] = position_polynomial.deriv()(
            time_stamps[-1]
        )  # Store last velocity

    return regressed_state


def detect_rebounds(
    time_stamps: ndarray,
    positions: ndarray,
    window_size: int,
    polynomial_degree: int = POLYNOMIAL_DEGREE,
    detection_threshold: float = DETECTION_THRESHOLD,
    table_height: float = TABLE_PLANE_HEIGHT,
) -> Union[List[int], Dict[int, ContactType]]:
    contact_dict = {}

    for i in range(window_size, len(positions)):
        positions_window = positions[i - window_size : i - 1, :]

        regressed_ball_state = get_regressed_state(
            time_stamps, positions_window, polynomial_degree
        )

        # Simulate the trajectory using the small model
        # ball_state: x, y, z, vx, vy, vz
        dt = time_stamps[i] - time_stamps[i - 1]
        simulated_ball_state = step_ball_simulation(regressed_ball_state, dt)

        # Calculate the Euclidean distance between the simulated trajectory and actual positions
        distance = norm(simulated_ball_state[:3] - positions[i])

        # Determine the contact type based on the rebound distance and table height
        if distance > detection_threshold:
            contact_type = ContactType.RACKET
        elif simulated_ball_state[2] > table_height:
            contact_type = ContactType.TABLE
        else:
            contact_type = ContactType.UNKNOWN

        # Add the rebound index and contact type to the respective lists
        contact_dict[i] = contact_type

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

    pass


def compute_racket_orientation(joint_angles_rad):
    quaternion = empty(4)
    normal_vector = empty(3)

    return quaternion, normal_vector


def calculate_rebound_velocity(V_initial, R_orient, C_normal):
    # Normalize vectors
    V_initial = np.array(V_initial) / np.linalg.norm(V_initial)
    C_normal = np.array(C_normal) / np.linalg.norm(C_normal)

    # Convert racket orientation to a unit quaternion
    R_quaternion = quat.from_float_array(R_orient)
    R_quaternion = quat.as_quat_array(R_quaternion / np.linalg.norm(R_quaternion))

    # Calculate reflection vector
    R_reflection = V_initial - 2 * np.dot(V_initial, C_normal) * C_normal

    # Transform reflection vector using quaternion rotation
    R_global = quat.as_float_array(R_quaternion * quat.as_quat_array(R_reflection) * quat.conjugate(R_quaternion))

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
