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


class Hayakawa2021RacketContact(BaseRacketContact):
    """
    Source:
    Hayakawa, Yoshikazu, et al.
    "Ball trajectory planning in serving task for table tennis
    robot."
    SICE Journal of Control, Measurement,
    and System Integration 9.2 (2016): 50-59.

    URL:
    https://www.tandfonline.com/doi/epdf/10.9746/jcmsi.9.50

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

    def forward(self, q) -> npt.NDArray:
        r_ball = 0.02  # ball radius
        mu = 0.25  # dynamic friction coefficient
        e_t = 0.93  # restituation coefficient

        v_z = np.abs(q[2])
        v_T = np.linalg.norm(q[0:3])

        v_s = 1 - 5 / 2 * mu * (1 + e_t) * v_z / v_T

        if v_T != 0 and v_s > 0:
            a = 0
        else:
            a = 2 / 5

        I_2 = np.array(
            [
                [1, 0],
                [0, 1],
            ]
        )
        S_12 = np.array(
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 0],
            ]
        )

        A_t = np.array([[(1 - a) * I_2, 0], [0, -e_t]])
        B_t = a * r_ball * S_12
        C_t = -(3 * a) / (2 * r_ball) * S_12
        D_t = np.array([[(1 - 3 * a / 2) * I_2, 1], [0, 1]])

        T = np.array([[A_t, B_t], [C_t, D_t]])

        return T @ q


class Nakashima2014RacketContact(BaseRacketContact):
    """
    Source:
    Nakashima, Akira, Daigo Ito, and Yoshikazu Hayakawa.
    "An online trajectory planning of struck ball with spin
    by table tennis robot."
    2014 IEEE/ASME International Conference on Advanced
    Intelligent Mechatronics. IEEE, 2014.

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6878188

    Ball Flight Model:      True
    Racket Rebound Model:   True
    Spin unit:              1/s or rad/s or None

    Approach:
    =============================
    Analytic model:         False
    System identification:  False
    ML model:               False
    Linear model:           False
    FEM:                    False
    Spring-Damper-Model:    False

    Effects considered:
    =============================
    Spin:                   False
    Friction:               False
    Ball elasticity:        False
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

    def forward():
        pass


class Liu2012RacketContact(BaseRacketContact):
    """
    Source:
    Liu, Chunfang, Yoshikazu Hayakawa, and Akira Nakashima.
    "Racket control and its experiments for robot playing table
    tennis."
    2012 IEEE International Conference on Robotics and
    Biomimetics (ROBIO).

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6490973

    Ball Flight Model:      True
    Racket Rebound Model:   True
    Spin unit:              1/s or rad/s or None

    Approach:
    =============================
    Analytic model:         False
    System identification:  False
    ML model:               False
    Linear model:           False
    FEM:                    False
    Spring-Damper-Model:    False

    Effects considered:
    =============================
    Spin:                   False
    Friction:               False
    Ball elasticity:        False
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

    def forward(self):
        pass


class Liu2013RacketContact(BaseRacketContact):
    """
    Source:
    Liu, Chunfang, Yoshikazu Hayakawa, and Akira Nakashima.
    "Racket control for a table tennis robot to return a ball."
    SICE Journal of Control, Measurement, and System
    Integration 6.4 (2013): 259-266.

    URL:
    https://www.tandfonline.com/doi/epdf/10.9746/jcmsi.6.259

    Ball Flight Model:      True
    Racket Rebound Model:   True
    Spin unit:              1/s or rad/s or None

    Approach:
    =============================
    Analytic model:         False
    System identification:  False
    ML model:               False
    Linear model:           False
    FEM:                    False
    Spring-Damper-Model:    False

    Effects considered:
    =============================
    Spin:                   False
    Friction:               False
    Ball elasticity:        False
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

    def forward(self):
        pass
