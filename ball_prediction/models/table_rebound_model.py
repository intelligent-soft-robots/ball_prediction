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
        T = np.array([[]])

        return q


class Nakashima2014TableContact(BaseTableContact):
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

    def forward():
        pass


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
        pass

    def forward(self):
        pass


class Liu2012TableContact(BaseTableContact):
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


class Liu2013TableContact(BaseTableContact):
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
        pass

    def forward(self):
        pass


class Zhang2010TableContactModel(BaseTableContact):
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
        pass

    def forward(self):
        pass


class ZZhang2010TableContactModel(BaseTableContact):
    """
    Source:
    Zhang, Zhengtao, De Xu, and Ping Yang.
    "Rebound model of table tennis ball for trajectory prediction."
    2010 IEEE International Conference on Robotics
    and Biomimetics. IEEE, 2010

    URL:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5723356

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
        pass

    def forward(self):
        pass


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

    def forward(self):
        pass


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
