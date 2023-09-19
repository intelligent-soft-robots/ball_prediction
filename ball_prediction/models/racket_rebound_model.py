from abc import ABC
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

# input: v_ball, F_magnusball, orientierung_racket, v_racket
# output: v_ball, F_magnusball


class BaseRacketContact(ABC):
    def __init__(self) -> None:
        raise NotImplementedError

    def forward(q_ball: Sequence[float], q_racket: Sequence[float]) -> Sequence[float]:
        pass


class SimpleRacketContact(BaseRacketContact):
    def __init__(self) -> None:
        self.restitution_factor = 0.97

    def forward(
        self, q_ball: Sequence[float], q_racket: Sequence[float]
    ) -> Sequence[float]:
        mu = self.restitution_factor

        contact_matrix = np.array(
            [
                [1.0000, 0.0000, 0.0000, 0.0000, 0.0015, 0.0000],
                [0.0000, 1.0000, 0.0000, 0.0015, 0.0000, 0.0000],
                [0.0000, 0.0000, -1.0 * mu, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
            ]
        )

        return contact_matrix @ q_ball


class Nakashima2010RacketContact(BaseRacketContact):
    """
    Source:
    Nakashima, Akira, et al.
    "Modeling of rebound phenomenon of a rigid ball with friction
    and elastic effects."
    Proceedings of the 2010 American Control Conference.

    URL:
    https://ieeexplore.ieee.org/document/5530520
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5530520

    Ball Flight Model:      True
    Racket Rebound Model:   True
    Spin unit:              1/s or rad/s or None

    Approach:
    =============================
    Analytic model:         False
    System identification:  False
    ML model:               False
    Linear model:           True
    FEM:                    False
    Spring-Damper-Model:    False

    Effects considered:
    =============================
    Spin:                   True
    Friction:               False
    Ball elasticity:        False- 5, -10, -20, -40, - 80
    Surface Rolling:        False
    Surface Unevenness:     False
    Point Contact:          False
    Surface Contact:        False

    Other assumptions or restrictions:
    -

    Args:
        BaseTableContact (abc): BaseTableContact default
        class.
    """

    def __init__(self) -> None:
        pass

    def forward(
        self,
        q_ball: Sequence[float],
        v_racket: Sequence[float],
        orientation_racket: Sequence[float],
    ) -> npt.NDArray:
        v_x = q_ball[0] - v_racket[0]
        v_y = q_ball[1] - v_racket[1]
        v_z = q_ball[2] - v_racket[2]

        omega_x = q_ball[3]
        omega_y = q_ball[4]
        omega_z = q_ball[5]

        q_ball_before = np.array([v_x, v_y, v_z, omega_x, omega_y, omega_z])

        alpha = orientation_racket[0]
        beta = orientation_racket[1]

        m_ball = 0.0027
        r_ball = 0.02  # ball radius
        mu = 0.25  # dynamic friction coefficient
        k_omega = 0.0
        k_v = 0.0
        e_r = 0.93  # restituation coefficient

        inertia_coeff = 2 / 3 * m_ball * r_ball**2

        R_R = np.array(
            [
                [
                    np.cos(beta),
                    np.sin(beta) * np.sin(alpha),
                    np.sin(beta) * np.cos(alpha),
                ],
                [0, np.cos(alpha), -np.sin(alpha)],
                [
                    -np.sin(beta),
                    np.cos(beta) * np.sin(alpha),
                    np.cos(beta) * np.cos(alpha),
                ],
            ]
        )

        block_upper = np.hstack((R_R, np.zeros(3)))
        block_lower = np.hstack((np.zeros(3), R_R))
        block = np.vstack((block_upper, block_lower))

        R = np.eye(6) - block

        # Inertia matrix in 2D
        S_12 = np.array(
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 0],
            ]
        )

        A_t = np.array(
            [
                [(1 - k_v) * inertia_coeff, 0, 0],
                [0, (1 - k_v) * inertia_coeff, 0],
                [0, 0, -e_r],
            ]
        )
        B_t = k_v * r_ball * S_12
        C_t = -k_omega / r_ball * S_12
        D_t = np.array(
            [
                [(1 - k_omega) * inertia_coeff, 0, 1],
                [0, (1 - k_omega) * inertia_coeff, 0],
                [0, 0, 1],
            ]
        )

        T_upper = np.hstack((A_t, B_t))
        T_lower = np.hstack((C_t, D_t))
        T = np.vstack((T_upper, T_lower))

        q_ball_after = R @ T @ R.T @ q_ball_before

        q_ball_after[0] = q_ball_after[0] + v_racket[0]
        q_ball_after[1] = q_ball_after[1] + v_racket[1]
        q_ball_after[2] = q_ball_after[2] + v_racket[2]

        return q_ball_after
