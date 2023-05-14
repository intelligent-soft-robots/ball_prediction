from typing import Optional, Sequence, Tuple

from ball_models import BallTrajectory
from numpy import array, diag, ndarray

from ball_prediction.ekf import ExtendedKalmanFilter


class BallTrajectoryEKF(ExtendedKalmanFilter):
    """Extends generic ExtendedKalmanFilter to the application
    of the passive dynamics of table tennis balls.

    Args:
        ExtendedKalmanFilter (ExtendedKalmanFilter): Standard
        ExtendedKalmanFilter class
    """

    def __init__(
        self, config: dict, ball_model: Optional[BallTrajectory] = None
    ) -> None:
        """Initialises EKF class.

        Args:
            config (dict): Configuration dict with configuration parameters.
            ball_model (BallTrajectory, optional): BallTrajectory class. Defaults to None.
        """
        dim_q = 9
        dim_z = 6
        dim_u = 0

        super().__init__(dim_q=dim_q, dim_z=dim_z, dim_u=dim_u)

        # Plug in ball dynamics for state transition calculation
        if ball_model is None:
            self.ball_model = BallTrajectory(config)
        else:
            self.ball_model = ball_model

        self.FJacobian = self.ball_model.compute_jacobian

        # Uncertainty matrices for EKF update
        self.Q = diag(config["ekf_param"]["Q"])
        self.R = diag(config["ekf_param"]["R"])

        self.P_init = diag(config["ekf_param"]["P_init_diag"])

    def initialize_kalman(
        self, q_init: Sequence[float], P_init: Optional[ndarray] = None
    ) -> None:
        """Initializes EKF by providing initial parameters.

        Args:
            q_init (Sequence[float]): Initial state.
            P_init (Optional[ndarray], optional): Initial state covariance
            matrix. Defaults to None.
        """
        self.q = array(q_init)

        if P_init is None:
            self.P = self.P_init
        else:
            self.P = P_init
            self.P_init = P_init

    def reset_P(self):
        """Utility function to reset covariance matrix P to initial
        coveriance matrix P.
        """
        self.ekf.P = self.P_init

    def set_Q(self, Q: ndarray):
        """Utility function to set the process uncertainty.

        Args:
            Q (ndarray): Process uncertainty.
        """
        self.Q = Q

    def set_R(self, R: ndarray):
        """Utility function to set measurement uncertainty.

        Args:
            R (ndarray): Measurement uncertainty.
        """
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

    def predict_update(
        self, z: Sequence[float], dt: float
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Convenience function for calling prediction and update step.

        Args:
            z (Sequence[float]): new measurement.
            dt (float): time step.

        Returns:
            Tuple[ndarray, ndarray, ndarray, ndarray]: returns predicted
            and estimated state and the corresponding new covariance
            matrices.
        """
        q_pred, P_pred = self.predict(dt)
        q_est, P_est = self.update(z)

        return q_pred, q_est, P_pred, P_est

    def HJacobian(self, q: Sequence[float]) -> ndarray:
        """Jacobian of ball trajectory measurement model with position and
        velocity directly provided by measurements.

        Args:
            q (Sequence[float]): State vector. For this model unused.

        Returns:
            ndarray: Jacobian of ball trajectory measurement model
        """
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

    def hq(self, q: Sequence[float]) -> ndarray:
        """Measurement model of ball trajectory with position and velocity
        directly provided by measurements.

        Args:
            q (Sequence[float]): State vector.

        Returns:
            ndarray: Measurement calculated by measurement model of state.
        """
        q = array(q)

        return q[0:6]
