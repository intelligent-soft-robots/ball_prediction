from typing import Sequence

from numpy import array, gradient, hstack
from numpy.polynomial.polynomial import Polynomial


def finite_differences_estimator(t_buffer: Sequence[float], z_buffer: Sequence[float]):
    time_stamps = array(t_buffer)
    positions = array(z_buffer)

    velocities = []

    for axs in range(3):
        velocity = gradient(positions[:, axs], time_stamps)
        velocities.append(velocity)

    velocities = array(velocities)
    print(positions)
    print(velocities)
    q_init = hstack((positions[-1], velocities))

    return array(q_init)


def regression_estimator(t_buffer: Sequence[float], z_buffer: Sequence[float], config):
    time_stamps = array(t_buffer)
    positions = array(z_buffer)

    degree = config.initial_state_estimation.regression_degree

    q_init = []

    positions = []
    velocities = []

    for axis in range(3):
        position_polynomial = Polynomial.fit(
            time_stamps[:-1], positions[:-1, axis], deg=degree
        )

        velocity_polynomial = position_polynomial.deriv()

        positions.append(position_polynomial(time_stamps[-1]))
        velocities.append(velocity_polynomial(time_stamps[-1]))

    spin = [0.0, 0.0, 0.0]

    q_init.extend(positions)
    q_init.extend(velocities)
    q_init.extend(spin)

    return array(q_init)


def external_measurement(t_buffer: Sequence[float], z_buffer: Sequence[float]):
    return array(z_buffer[-1])


def external_measurement_wo_spin(t_buffer: Sequence[float], z_buffer: Sequence[float]):
    q_init = []
    q_init.extend(z_buffer[-1])
    q_init.extend([0.0, 0.0, 0.0])

    return array(q_init)

def launcher(launch_parameter, t_buffer, z_buffer):
    pass