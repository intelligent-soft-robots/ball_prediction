from typing import Optional, Sequence

import ball_models
from numpy import array, diag, ndarray

from ball_prediction.ekf import ExtendedKalmanFilter


class BallTrajectoryEKF(ExtendedKalmanFilter):
    def __init__(self, config, ball_model=None) -> None:
        dim_q = 9
        dim_z = 6
        dim_u = 0

        super().__init__(dim_q=dim_q, dim_z=dim_z, dim_u=dim_u)

        # Plug in ball dynamics for state transition calculation
        if ball_model is None:
            self.ball_model = ball_models.BallTrajectory(config)
        else:
            self.ball_model = ball_model

        self.FJacobian = self.ball_model.compute_jacobian()

        # Uncertainty matrices for EKF update
        self.Q = diag(config.ekf_param.Q)
        self.R = diag(config.ekf_param.R)

        self.P_init = diag(config.ekf_param.P_init_diag)

    def initialize_kalman(
        self, q_init: Sequence[float], P_init: Optional[ndarray] = None
    ) -> None:
        self.q = array(q_init)

        if P_init is None:
            self.P = self.P_init
        else:
            self.P = P_init
            self.P_init = P_init

    def reset_P(self):
        self.ekf.P = self.P_init

    def set_Q(self, Q: ndarray):
        self.Q = Q

    def set_R(self, R: ndarray):
        self.R = R

    def predict_state(self, dt: float, q: Optional[ndarray] = None) -> ndarray:
        """Overwrites generic predict state method from EKF with ball

        Args:
            dt (float): Time step

        Returns:
            numpy.ndarray: State vector
        """
        if q is None:
            q = self.q

        return self.ball_model.integrate(q, dt)

    def predict_update(self, z, dt):
        q_pred, P_pred = self.predict(dt)
        q_est, P_est = self.update(z)

        return q_pred, q_est, P_pred, P_est

    def measurement_model():

        return True

    def HJacobian():
        HJacobian = array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

        return HJacobian

    def hq(q: Sequence[float]):
        q = array(q)
        z = q[0:6]

        return z
