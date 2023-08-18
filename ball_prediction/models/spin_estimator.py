import logging
import warnings

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


from ball_prediction.models.racket_kinematics import compute_racket_orientation
from ball_prediction.models.ball_simulation import BallSimulationSpin
from ball_prediction.models.utils import ContactType
from ball_prediction.models.magnus_regressor import compute_velocity_regression



def linear_table_contact(velocity_before_bounce):
    # assuming there is no friction effect, point contact

    velocity_after_bounce = velocity_before_bounce
    velocity_after_bounce = -velocity_after_bounce[2]
    return velocity_after_bounce


def lineare_racket_contact(
    velocity_before_bounce: Sequence[float], robot_joint_angles_rad: Sequence[float]
):
    # ! Assumptions:
    # Point-contact of ball and surface, ball is not spinning,
    # ball does not lose energy due to friction or slip
    # ball and racket does not lose energy due to elastic deformation of the ball
    # all energy is fully maintained.
    # surface is perfectly flat and is fully described by normal vector.
    
    racket_normal = compute_racket_orientation(joint_angles_rad=robot_joint_angles_rad)
    
    v_before = array(velocity_before_bounce)
    racket_normal = array(racket_normal)

    # Normalize vectors
    racket_normal = racket_normal / norm(racket_normal)

    # Calculate the component of v_in that is perpendicular to the surface
    v_perpendicular = (v_before @ racket_normal) * racket_normal

    # Calculate the component of v_in that is parallel to the surface
    v_parallel = v_before - v_perpendicular

    # Calculate the velocity vector of the ball after the collision (v_out)
    velocity_after_bounce =  v_parallel - v_perpendicular

    return velocity_after_bounce


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

            vel_before_bounce = compute_velocity_regression(
                time_stamps_before_bounce, positions_before_bounce
            )
            vel_before_bounce = vel_before_bounce[-1]

            # Regression after bounce
            time_stamps_after_bounce = time_stamps[
                index + 1 : index + n_regression_samples
            ]
            positions_after_bounce = positions[index + 1 : index + n_regression_samples]

            vel_after_bounce = compute_velocity_regression(
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
