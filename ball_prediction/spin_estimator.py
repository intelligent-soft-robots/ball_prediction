import warnings
from typing import Dict, List, Sequence, Tuple

from numpy import abs, arctan2, array, hstack, meshgrid, ndarray, polynomial, sqrt, empty_like
from scipy.signal import find_peaks

TABLE_PLANE_HEIGHT = 0.77
TABLE_DETECTION_THRESHOLD = -0.03
TABLE_DETECTION_SAMPLE_DISTANCE = 10

def detect_racket_contact(
    time_stamps: ndarray,
    positions: ndarray,
):
    # we can apply here heavy filtering since we do not want to find
    # peaks but changes of trends.


    # make a regression of the next 10 ball positions on basis of the last 10
    # ball positions. If there is a significant change in direction it is either
    # noise (position exceeds plausility threshold) or a contact

    # contact_specification on? if yes, it requires a height and optionally a region
    # 
    pass
    


def detect_table_contact(
    positions: ndarray, 
    table_height: float = TABLE_PLANE_HEIGHT,
    detection_threshold: float = TABLE_DETECTION_THRESHOLD,
    sample_distance: float = TABLE_DETECTION_SAMPLE_DISTANCE
):
    positions = array(positions)
    positions_axis = positions[:, 2]

    positions_axis -= table_height
    positions_axis *= -1

    table_contact_indices = find_peaks(
        positions_axis,
        height= detection_threshold,
        distance= sample_distance,
    )[0]

    return table_contact_indices


def detect_rebounds() -> List[int]:
    pass

    # detect bounce indices


def linear_table_contact(velocity_before_bounce):
    # assuming there is no friction effect, point contact

    velocity_after_bounce = velocity_before_bounce
    velocity_after_bounce = -velocity_after_bounce[2]
    return velocity_after_bounce

def lineare_racket_contact(
    velocity_before_bounce: Sequence[float], 
    racket_orientation: Sequence[float]
):
    # assume no friction, point contact, no restitution, ball hits racket on flat surface,
    # surface is even


    
    pass


def check_difference_below_threshold(indices, threshold):
    # Convert the list to a NumPy array for faster computations
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
                warnings.warn(f"Combination values below threshold: {indices[i]}, {indices[j]}")
                return True

    # If no combination satisfies the condition, return False
    return False

def velocity_regression(
    time_stamps: ndarray,
    positions: ndarray,
    degree: int = 3,
) -> Sequence[float]:
    velocities = []

    for axis in range(3):
        velocity = []

        position_polynomial = polynomial.polynomial.Polynomial.fit(
            time_stamps, positions[:, axis], deg=degree
        )

        velocity_polynomial = position_polynomial.deriv()

        for t in time_stamps:
            velocity.append(velocity_polynomial(t))

        velocities.append(velocity)

    return hstack(velocities)


def estimate(
    time_stamps: Sequence[float],
    positions: Sequence[Sequence[float]], 
    velocities: Sequence[Sequence[float]],
    regression: bool = True,
    n_regression_samples: int = 10,
    use_invertable_contact_model: bool = False,
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
    rebound_indices = detect_rebounds(positions)

    check_difference_below_threshold(rebound_indices, n_regression_samples)

    # take velocity vector before bounce and after bounce
    for index in rebound_indices:
        if regression:
            time_stamps_before_bounce = time_stamps[index-n_regression_samples:index]
            positions_before_bounce = positions[index-n_regression_samples:index]

            vel_before_bounce = velocity_regression(time_stamps_before_bounce, positions_before_bounce)
            vel_before_bounce = vel_before_bounce[-1]

            time_stamps_after_bounce = time_stamps[index+1: index+n_regression_samples]
            positions_after_bounce = positions[index+1: index+n_regression_samples]
            
            vel_after_bounce = velocity_regression(time_stamps_after_bounce, positions_after_bounce)
            vel_after_bounce = vel_after_bounce[0]

        else:      
            vel_before_bounce = velocities[index-1]
            vel_after_bounce = velocities[index+1]

        # only works for table contacts...
        # TODO: what to do with racket contacts? we need the racket orientation! 
        # if contact_type == "table"
        vel_after_bounce_no_spin = linear_table_contact(vel_before_bounce)

        # if contact_type == "racket"


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
