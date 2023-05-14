import cProfile as profile
from typing import Optional, Sequence, Tuple

import ball_models
from numpy import arange, diag, eye, ndarray, std

from ball_prediction.initial_state_estimator import InitialStateEstimator
from ball_prediction.prediction_filter import PredictionFilter
from ball_prediction.trajectory_ekf import BallTrajectoryEKF


class TrajectoryPredictor(BallTrajectoryEKF):
    """Extends BallTrajectoryEKF with utility functions enabling usage
    of EKF in lab environments.

    Args:
        BallTrajectoryEKF (BallTrajectoryEKF): Generic EKF extended with
        ball trajectory model.
    """

    def __init__(self, config: dict) -> None:
        """Initialises TrajectoryPredictor.

        Args:
            config (dict): Configuration dict storing prediction parameters.
        """
        self.t_prediction_horizon = config["setting"]["t_prediction_horizon"]
        self.f_predictor = config["setting"]["f_predictor"]
        self.init_buffer_size = config["setting"]["init_buffer_size"]
        self.dynamic_R = config["setting"]["dynamic_R"]

        self.prediction_filter = PredictionFilter.load_filter(config)
        self.inital_state_estimator = InitialStateEstimator.load_estimator(config)

        config_path = "/home/adittrich/test_workspace/workspace/src/ball_models/config/config.toml"
        self.ball_model = ball_models.BallTrajectory(config_path)

        super().__init__(config=config, ball_model=self.ball_model)

        # Storage
        self.t_current = 0.0

        self.t = []
        self.z = []

        self.q_preds = []
        self.q_ests = []

    def initialize_predictor(
        self,
        q_init: Sequence[float],
        P_init: Optional[ndarray] = None,
    ) -> None:
        """Initializes predictor by initialising EKF and storing initial covariance
        matrix if reset is triggered.

        Args:
            q_init (Sequence[float]): Initial state
            P_init (Optional[ndarray], optional): Initial covariance matrix.
            Defaults to None.
        """
        if P_init is None:
            P_init = eye(len(q_init))

        self.initialize_kalman(q_init, P_init)

    def predict_horizon(self) -> None:
        """Predicts the future ball states on basis of current estimate of
        predictor.
        """
        self.reset_predictions()

        q = self.q.copy()

        if len(self.t) >= 2:
            t_current = self.t[-1] - self.t[0]
            duration = self.t_prediction_horizon - t_current
            dt = 1 / self.f_predictor

            self.t_pred = arange(t_current, self.t_prediction_horizon, dt)
            self.q_pred = self.ball_model.simulate(q, duration, dt)

    def kalman_update_step(self) -> None:
        """Performs kalman update step with latest measurement in measurement
        storage.
        """
        dt = self.t[-1] - self.t[-2]
        q_pred, q_est, P_pred, P_est = self.predict_update(self.z[-1], dt)

        self.q_preds.append(q_pred)
        self.q_ests.append(q_est)

    def reset_predictions(self) -> None:
        """Resets predictions and covariance matrix."""
        self.t_pred = []
        self.q_pred = []
        self.P = self.P_init

    def reset(self) -> None:
        """Resets predictor by removing all stored samples."""
        self.t = []
        self.z = []

        self.reset_P()

    def input_samples(self, z: Sequence[float], time_stamp: float) -> None:
        """Receives new measurement and time stamp of the measurement and stores
        it in the sample storage. From the sample storage future ball states can
        be predicted and the predicted state can be updated via the kalman update
        step.

        Args:
            z (Sequence[float]): Measurement.
            time_stamp (float): Time step of the measurement.
        """
        self.t.append(time_stamp)
        self.z.append(z)

        if self.init_buffer_size < len(self.z):
            self.kalman_update_step()
            self.predict_horizon()

        if self.init_buffer_size == len(self.z):
            q_init = self.inital_state_estimator.estimate_extended_state_space(
                self.t, self.z
            )
            self.initialize_predictor(q_init=q_init)

            if self.dynamic_R:
                R = diag(std(self.z, axis=0))
                self.set_R(R)

        self.t_current = time_stamp

    def get_prediction(
        self, filter: bool = True
    ) -> Tuple[Sequence[float], Sequence[float]]:
        """Returns predictions from current predicted state.

        Args:
            filter (bool, optional): For debugging purposes also unfiltered 
            predictions can be returned. 
            This can be specified via the filter argument. Defaults to True.

        Returns:
            Tuple[Sequence[float], Sequence[float]]: Filtered prediction 
            according to parameters set in configuration file.
        """
        t_pred = self.t_pred
        q_pred = self.q_pred

        if len(q_pred) == 0:
            return [], []

        if filter:
            t_pred, q_pred = self.prediction_filter.filter(t_pred, q_pred)

        return t_pred, q_pred

    def profile_predict(self) -> None:
        profile.runctx("self.predict_horizont()", globals(), locals())

    def profile_update(self, z, t) -> None:
        profile.runctx("self.kalman_update_step(z, t)", globals(), locals())
