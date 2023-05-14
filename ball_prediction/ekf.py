from typing import Tuple

from numpy import array, eye, ndarray, zeros
from numpy.linalg import inv


class ExtendedKalmanFilter:
    def __init__(self, dim_q: int, dim_z: int, dim_u: int = 0) -> None:
        """Initialisation of barebone kalman filter

        Args:
            dim_q (int): Dimensionality of state.
            dim_z (int): Dimensionality of measurements
            dim_u (int, optional): Dimensionality of control signal.
            Defaults to 0 for a unactuated passive system.
        """
        self.dim_q = dim_q
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.q = zeros((dim_q, 1))  # state
        self.P = eye(dim_q)  # uncertainty covariance

        self.y = zeros((dim_z, 1))  # residual

        # Process and measurement uncertainty
        self.Q = eye(dim_q)
        self.R = eye(dim_z)

        self.F = eye(dim_q)
        self.B = 0
        self.I = eye(dim_q)

    def predict_state(self, u: ndarray, dt: float) -> ndarray:
        """Needs to be overwritten for specific system"""
        q = self.q

        F = self.F
        B = self.B

        return F @ q + B @ u

    def predict(self, dt: float) -> Tuple[ndarray, ndarray]:
        """Prediction step of Extended Kalman Filter. Takes estimated
        state and given state transition model specified by predict_state
        and FJacobian and predicts future state.

        Args:
            dt (float): time step

        Returns:
            Tuple[ndarray, ndarray]: Predicted state vector and predicted
            Covariance matrix.
        """
        # Fetch variables
        q = self.q
        P = self.P
        Q = self.Q

        q = self.predict_state(dt)
        F = self.FJacobian(q)

        # compute error covariance ahead
        P = F @ P @ F.T + Q

        # update state and covariance
        self.q = q
        self.P = P

        return q, P

    def update(self, z: ndarray) -> Tuple[ndarray, ndarray]:
        """Updates current prediction with kalman estimate step.

        Args:
            z (ndarray): latest measurement complying the measurement
            model specified with the functions HJacobian and hq

        Returns:
            Tuple[ndarray, ndarray]: Estimated state vector and
            estimated Covariance matrix.
        """
        z = array(z)
        q = self.q

        # Fetch variables for better readability
        P = self.P
        R = self.R
        H = self.HJacobian(q)

        # compute kalman gain
        PHT = P @ H.T
        S = H @ PHT + R
        S_inv = inv(S)
        K = PHT @ S_inv

        y = z - self.hq(q)
        Ky = K @ y
        q = q + Ky

        # compute error covariance
        I_KH = self.I - K @ H
        P = I_KH @ P @ I_KH.T + K @ R @ K.T

        # overwrite internal state
        self.q = q
        self.P = P

        return q, P
