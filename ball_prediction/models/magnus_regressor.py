from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

PHYSICS_CFG = {
    "ball_mass": 0.027,
    "ball_radius": 0.02,
    "air_density": 1.18,
    "gravitational_constant": 9.80801,
    "drag_coefficient": 0.47016899,
    "lift_coefficient": 1.46968343,
    "decay_coefficient": 0.005054473513775579,
}

REGRESS_CFG = {
    "polynomial_degree": 3,
}

TimeTrajectory = Sequence[float]
PositionTrajectory = Sequence[Tuple[float]]


Velocity3D = npt.NDArray
Acceleration3D = npt.NDArray
Force3D = npt.NDArray


class MagnusRegressor:
    def __init__(self, regression_config, physics_config) -> None:
        self.poly_deg = regression_config["polynomial_degree"]

        self.m_ball = physics_config["ball_mass"]
        self.r_ball = physics_config["ball_radius"]
        self.rho = physics_config["air_density"]
        self.g = physics_config["gravitational_constant"]
        self.c_drag = physics_config["drag_coefficient"]
        self.c_lift = physics_config["lift_coefficient"]
        self.c_decay = physics_config["decay_coefficient"]

        self.A = np.pi * self.r_ball**2

    def _derivate(
        self,
        time_stamps: np.typing.NDArray,
        positions: np.typing.NDArray,
        eval_time: Optional[float] = None,
    ) -> Tuple[Velocity3D, Acceleration3D]:
        v = []
        a = []

        for axis in range(3):
            poly = np.polynomial.polynomial.Polynomial.fit(
                time_stamps, positions[:, axis], deg=self.poly_deg
            )
            velocities = poly.deriv()
            accelerations = poly.deriv(2)

            v.append(velocities(eval_time))
            a.append(accelerations(eval_time))

        return np.array(v), np.array(a)

    def _compute_drag_force(self, v: Velocity3D) -> Force3D:
        c_drag = self.c_drag
        rho = self.rho
        A = self.A

        return -0.5 * rho * c_drag * A * np.linalg.norm(v) * v

    def _compute_gravity_force(self) -> Force3D:
        m_ball = self.m_ball
        g = self.g
        return -m_ball * g * np.array([0.0, 0.0, 1.0])

    def compute(
        self,
        time_stamps: TimeTrajectory,
        positions: PositionTrajectory,
        eval_time: float,
    ) -> Tuple[float]:
        time_stamps = np.array(time_stamps)
        positions = np.array(positions)

        if eval_time is not None:
            v_ball, a_ball = self._derivate(
                time_stamps=time_stamps, positions=positions, eval_time=eval_time
            )

        m_ball = self.m_ball

        F_d = self._compute_drag_force(v_ball)
        F_g = self._compute_gravity_force()

        return m_ball * a_ball - F_g - F_d


def compute_magnus_force(velocity, spin, physics_config):
    v = np.array(velocity)
    omega = np.array(spin)

    r_ball = physics_config["ball_radius"]
    rho = physics_config["air_density"]
    c_lift = physics_config["lift_coefficient"]
    A = np.pi * r_ball**2

    return 0.5 * rho * c_lift * A * r_ball * np.cross(omega, v)


def compute_velocity_regression(
    time_stamps: np.ndarray,
    positions: np.ndarray,
    polynomial_degree: int = 3,
) -> np.ndarray:
    velocities = np.empty_like(positions)

    for axis in range(3):
        position_polynomial = np.polynomial.Polynomial.fit(
            time_stamps, positions[:, axis], deg=polynomial_degree
        )

        velocity_polynomial = position_polynomial.deriv()

        velocities[:, axis] = velocity_polynomial(time_stamps)

    return velocities
