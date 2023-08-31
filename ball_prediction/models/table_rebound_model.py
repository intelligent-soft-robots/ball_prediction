# input: v_ball, F_magnusball
# output: v_ball, F_magnusball

from abc import ABC
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt


class BaseTableContact(ABC):
    def __init__(self) -> None:
        raise NotImplementedError

    def forward(q: Sequence[float]) -> Sequence[float]:
        pass


class SimpleTableContact(BaseTableContact):
    def __init__(self) -> None:
        self.restitution_factor = 0.97

    def forward(self, q):
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

        return contact_matrix @ q


class Hayakawa2021TableContact(BaseTableContact):
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
        m_ball = 0.0027  # ball mass
        r_ball = 0.02  # ball radius
        mu = 0.25  # dynamic friction coefficient
        e_t = 0.93  # restituation coefficient

        v_x = q[0]
        v_y = q[1]
        v_z = q[2]
        omega_x = q[3]
        omega_y = q[4]
        omega_z = q[5]

        v_T = np.hstack((v_x - r_ball * omega_y, v_y + r_ball * omega_x, 0))
        v_T_norm = np.linalg.norm(v_T)

        v_s = 1 - 5 / 2 * mu * (1 + e_t) * v_z / v_T_norm

        if v_T_norm != 0 and v_s > 0:
            a = mu * (1 + e_t) * np.abs(v_z) / v_T_norm
        else:
            a = 2 / 5

        inertia_coeff = 2 / 3 * m_ball * r_ball**2

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
                [(1 - a) * inertia_coeff, 0, 0],
                [0, (1 - a) * inertia_coeff, 0],
                [0, 0, -e_t],
            ]
        )
        B_t = a * r_ball * S_12
        C_t = -(3 * a) / (2 * r_ball) * S_12
        D_t = np.array(
            [
                [(1 - 3 * a / 2) * inertia_coeff, 0, 1],
                [0, (1 - 3 * a / 2) * inertia_coeff, 0],
                [0, 0, 1],
            ]
        )

        T_upper = np.hstack((A_t, B_t))
        T_lower = np.hstack((C_t, D_t))
        T = np.vstack((T_upper, T_lower))

        return T @ q


class Zhang2014TableContact(BaseTableContact):
    """
    Source:
    Zhang, Yifeng, et al.
    "Spin observation and trajectory prediction of a ping-pong ball."
    2014 IEEE international conference on robotics and
    automation (ICRA).

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907456

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

    def forward(self, q: Sequence[float]) -> Sequence[float]:
        v_x = q[0]
        v_y = q[1]
        v_z = q[2]

        # Paper uses rad/s. Transform 1/s to rad/s.
        omega_x = q[3] * 2 * np.pi
        omega_y = q[4] * 2 * np.pi
        omega_z = q[5] * 2 * np.pi

        b_1 = np.array([0.75, 0.0015])
        b_2 = np.array([0.75, 0.0015])
        b_3 = np.array([-0.97])
        b_4 = np.array([-26.0, 0.53])
        b_5 = np.array([25.0, 0.6])
        b_6 = np.array([0.9])

        v_x_out = np.hstack([v_x, omega_y]) @ b_1
        v_y_out = np.hstack([v_y, omega_x]) @ b_2
        v_z_out = np.hstack([v_z]) @ b_3

        omega_x_out = np.hstack([v_y, omega_x]) @ b_4
        omega_y_out = np.hstack([v_x, omega_y]) @ b_5
        omega_z_out = np.hstack([omega_z]) @ b_6

        # Transform back to 1/s
        omega_x_out /= 2 * np.pi
        omega_y_out /= 2 * np.pi
        omega_z_out /= 2 * np.pi

        return np.hstack(
            (v_x_out, v_y_out, v_z_out, omega_x_out, omega_y_out, omega_z_out)
        )


class Nakashima2010TableContact(BaseTableContact):
    """
    Source:
    Nakashima, Akira, et al.
    "Modeling of rebound phenomenon of a rigid ball with friction
    and elastic effects."
    Proceedings of the 2010 American Control Conference.

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5530520

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

    def forward(self, q: Sequence[float]) -> Sequence[float]:
        v_x = q[0]
        v_y = q[1]
        v_z = q[2]
        omega_x = q[3]
        omega_y = q[4]
        omega_z = q[5]

        v_norm = np.linalg.norm(q[0:3])

        r_ball = 0.02
        mu = 0.25
        e_t = 0.93

        alpha = mu * (1 + e_t) * v_z / v_norm

        A_v = np.array(
            [
                [1 - alpha, 0, 0],
                [0, 1 - alpha, 0],
                [0, 0, -e_t],
            ]
        )

        B_v = np.array(
            [
                [0, alpha * r_ball, 0],
                [-alpha * r_ball, 0, 0],
                [0, 0, 0],
            ]
        )

        A_omega = np.array(
            [
                [0, -3 * alpha / (2 * r_ball), 0],
                [-3 * alpha / (2 * r_ball), 0, 0],
                [0, 0, 0],
            ]
        )

        B_omega = np.array(
            [
                [1 - 3 * alpha / 2, 0, 0],
                [0, 1 - 3 * alpha / 2, 0],
                [0, 0, 1],
            ]
        )

        T_upper = np.hstack((A_v, B_v))
        T_lower = np.hstack((A_omega, B_omega))
        T = np.vstack((T_upper, T_lower))

        return T @ q


class Huang2011TableContact(BaseTableContact):
    """
    Source:
    Huang, Yanlong, et al.
    "Trajectory prediction of spinning ball for ping-pong player
    robot."
    2011 IEEE/RSJ International Conference on Intelligent
    Robots and Systems.

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6095044

    Ball Flight Model:      True
    Racket Rebound Model:   False
    Spin unit:              rad/s

    Approach:
    =============================
    Analytic model:         False
    System identification:  True
    ML model:               False
    Linear model:           True
    FEM:                    False
    Spring-Damper-Model:    False

    Effects considered:
    =============================
    Spin:                   True
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

    def forward(self, q: Sequence[float]) -> Sequence[float]:
        v_x = q[0]
        v_y = q[1]
        v_z = q[2]

        # Paper uses rad/s. Transform 1/s to rad/s.
        omega_x = q[3] * 2 * np.pi
        omega_y = q[4] * 2 * np.pi
        omega_z = q[5] * 2 * np.pi

        b_1 = np.array([0.6278, -0.0003, -0.0344])
        b_2 = np.array([0.7796, 0.0011, 0.3273])
        b_3 = np.array([-0.5498, 0.8735])
        b_4 = np.array([7.4760, 0.1205, 39.4228])
        b_5 = np.array([-22.9295, 0.1838, -13.4791])
        b_6 = np.array([-0.3270, 39.9528])

        v_x_out = np.hstack([v_x, omega_y, 1]) @ b_1
        v_y_out = np.hstack([v_y, omega_x, 1]) @ b_2
        v_z_out = np.hstack([v_z, 1]) @ b_3
        omega_x_out = np.hstack([v_y, omega_x, 1]) @ b_4
        omega_y_out = np.hstack([v_x, omega_y, 1]) @ b_5
        omega_z_out = np.hstack([omega_z, 1]) @ b_6

        # Transform back to 1/s
        omega_x_out /= 2 * np.pi
        omega_y_out /= 2 * np.pi
        omega_z_out /= 2 * np.pi

        return np.hstack(
            (v_x_out, v_y_out, v_z_out, omega_x_out, omega_y_out, omega_z_out)
        )


class Zhang2010TableContact(BaseTableContact):
    """
    Source:
    Zhang, Zhengtao, De Xu, and Min Tan.
    "Visual measurement and prediction of ball trajectory for table
    tennis robot."
    IEEE Transactions on Instrumentation and Measurement
    59.12 (2010): 3195-3205.

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5454397

    Ball Flight Model:      True
    Racket Rebound Model:   True
    Spin unit:              None

    Approach:
    =============================
    Analytic model:         False
    System identification:  True
    ML model:               False
    Linear model:           True
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
        BaseTableContact ((abc): BaseTableContact default
        class.
    """

    def __init__(self) -> None:
        pass

    def forward(self, q: Sequence[float]) -> Sequence[float]:
        v_x = q[0]
        v_y = q[1]
        v_z = q[2]
        omega_x = q[3]
        omega_y = q[4]
        omega_z = q[5]

        K_rx = 0.50259
        K_ry = 0.75204
        K_rz = -0.87613

        b_x = 0.50652
        b_y = -0.011442
        b_z = 0.32194

        v_x_out = K_rx * v_x + b_x
        v_y_out = K_ry * v_y + b_y
        v_z_out = K_rz * v_z + b_z

        omega_x_out = omega_x
        omega_y_out = omega_y
        omega_z_out = omega_z

        return np.array(
            [v_x_out, v_y_out, v_z_out, omega_x_out, omega_y_out, omega_z_out]
        )


class Zhao2016TableContact(BaseTableContact):
    """
    Source:
    Zhao, Yongsheng, Rong Xiong, and Yifeng Zhang.
    "Rebound modeling of spinning ping-pong ball based on multiple
    visual measurements."
    IEEE Transactions on Instrumentation
    and Measurement 65.8 (2016): 1836-1846.

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7469824

    Ball Flight Model:      True
    Racket Rebound Model:   True
    Spin unit:              1/s or rad/s or None

    Approach:
    =============================
    Analytic model:         False
    System identification:  False
    ML model:               True
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

    def forward(self, q: Sequence[float]) -> Sequence[float]:
        T = np.array(
            [
                [1, 0, f_mu_x * 1],
            ]
        )


class Nonomura2010TableContact(BaseTableContact):
    """
    Source:
    Nonomura, Junko, Akira Nakashima, and Yoshikazu Hayakawa.
    "Analysis of effects of rebounds and aerodynamics for trajectory
    of table tennis ball."
    Proceedings of SICE Annual Conference 2010.

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5603024

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
        # See Nakashima, Akira, et al. 2010
        raise NotImplementedError


class Bao2012TableContact(BaseTableContact):
    """
    Source:
    Bao, Han, et al.
    "Bouncing model for the table tennis trajectory prediction and
    the strategy of hitting the ball."
    2012 IEEE International Conference on Mechatronics and
    Automation.

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6285129

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
        raise NotImplementedError

    def forward(self, q: Sequence[float]) -> Sequence[float]:
        pass


class ZZhang2010TableContactModel(BaseTableContact):
    """
    Source:
    Zhang, Zhengtao, De Xu, and Ping Yang.
    "Rebound model of table tennis ball for trajectory prediction."
    2010 IEEE International Conference on Robotics
    and Biomimetics. IEEE, 2010

    URL: Hayakawa2021TableContact()

    model.forward([1, 1, 1, 1, 1, 1])
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
        BaseTableContact ((abc): BaseTableContact default
        class.
    """

    def __init__(self) -> None:
        raise NotImplementedError


class MuJoCo2012TableContact(BaseTableContact):
    """
    Use MuJoCo simulation as baseline.

    Source
    Todrov, Emanuel, Tom Erez, and Yuval Tassa.
    "Mujoco: A physics engine for model-based control."
    2012 IEEE/RSJ international conference on intelligent
    robots and systems. IEEE, 2012.

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6386109

    Args:
        BaseTableContact (abc): BaseTableContact default
        class.
    """

    pass


class AnalyticTableContact(BaseTableContact):
    def __init__(self) -> None:
        pass

    def forward(self):
        pass


class MLPTableContact(BaseTableContact):
    def __init__(self, torch_file: str) -> None:
        pass

    def forward(self):
        pass


class PINNTableContact(BaseTableContact):
    def __init__(self, torch_file: str) -> None:
        pass

    def forward(self):
        pass


class GPTableContact(BaseTableContact):
    def __init__(self, kernel_file: str) -> None:
        pass

    def forward(self):
        pass


class ResTableContact(BaseTableContact):
    def __init__(self) -> None:
        pass

    def forward(self):
        pass


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    model = Hayakawa2021TableContact()
    model.forward([1, 1, 1, 1, 1, 1])
