import logging
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
        self.fixed_prediction_horizont = config["setting"]["fixed_prediction_horizont"]

        self.prediction_filter = PredictionFilter.load_filter(config)
        self.inital_state_estimator = InitialStateEstimator.load_estimator(config)

        config_path = "/home/adittrich/test_workspace/workspace/src/ball_models/config/config.toml"
        self.ball_model = ball_models.BallTrajectory(config_path)

        super().__init__(config=config, ball_model=self.ball_model)

        # Overall measurement storage
        self.t = []
        self.z = []

        # Simulated trajectory based on current estimated state
        self.t_simulated = []
        self.q_simulated = []

        # Estimated and predicted states by EKF
        self.q_predicted = []
        self.q_estimated = []

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
        print(f"INITIAL STATE: {q_init}")
        self.initialize_kalman(q_init, P_init)

    def predict_horizon(self) -> None:
        """Predicts the future ball states on basis of current estimate of
        predictor.

        By modifying fixed_prediction_horizont in the configuration file,
        the stored prediction length can be varied.

        For fixed_prediction_horizont equal to True, the overall prediction
        horizon is fixed until a specified time stamp. The later the prediction
        starts the smaller will be the prediction size.

        Forfixed_prediction_horizont to False, there will be a fixed number
        of predictions generated starting from current time_stamp.
        """
        self.reset_predictions()

        if self.init_buffer_size <= len(self.z):
            # State q is not intialized yet.
            # t_current requires at least 2 time stamps in t storage
            q = self.q.copy()
            dt = 1 / self.f_predictor

            if self.fixed_prediction_horizont:
                # fixed prediction horizon
                t_current = self.t[-1] - self.t[0]
                t_end = self.t_prediction_horizon
                duration = self.t_prediction_horizon - t_current + dt
            else:
                # fixed number of samples from current time_step
                t_current = self.t[-1] - self.t[0]
                t_end = t_current + self.t_prediction_horizon
                duration = self.t_prediction_horizon

            self.t_simulated = arange(t_current, t_end, dt)
            self.q_simulated = self.ball_model.simulate(q, duration, dt)
        else:
            logging.info(f"Initial estimate not performed yet!")

    def kalman_update_step(self) -> None:
        """Performs kalman update step with latest measurement in measurement
        storage.
        """
        if len(self.t) > 1:
            dt = self.t[-1] - self.t[-2]

            q_pred, q_est, P_pred, P_est = self.predict_update(self.z[-1], dt)

            self.q_predicted.append(q_pred)
            self.q_estimated.append(q_est)

    def reset_predictions(self) -> None:
        """Resets predictions and covariance matrix."""
        self.t_simulated = []
        self.q_simulated = []

    def reset_ekf(self) -> None:
        """Resets predictor by removing all stored samples."""
        self.t = []
        self.z = []

        self.reset_predictions()
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

        # initial buffer size should be minimum 2 samples

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
        t_simulated = self.t_simulated
        q_simulated = self.q_simulated

        if len(q_simulated) == 0:
            return [], []

        if len(t_simulated) != len(q_simulated):
            logging.warning(
                f"Length of time {len(t_simulated)} and "
                f"states {len(q_simulated)} do not match!"
            )
            return [], []

        if filter and len(q_simulated) > 10:
            # filter methods require some samples to work efficiently
            t_simulated, q_simulated = self.prediction_filter.filter(
                t_simulated, q_simulated
            )

        return t_simulated, q_simulated
