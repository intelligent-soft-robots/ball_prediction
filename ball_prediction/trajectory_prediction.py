import cProfile as profile
from typing import Optional, Sequence, Union

import ball_models
from numpy import arange, diag, eye, ndarray, std

from ball_prediction.initial_state_estimator import InitialStateEstimator
from ball_prediction.prediction_filter import PredictionFilter
from ball_prediction.trajectory_ekf import BallTrajectoryEKF


class TrajectoryPredictor(BallTrajectoryEKF):
    def __init__(
        self,
        config: dict,
    ) -> None:
        self.t_prediction_horizon = config["setting"]["t_prediction_horizon"]
        self.f_predictor = config["setting"]["f_predictor"]
        self.init_buffer_size = config["setting"]["init_buffer_size"]
        self.prediction_filter = PredictionFilter(
            config["setting"]["prediction_filter"]
        )
        self.inital_state_estimator = InitialStateEstimator(
            config["setting"]["initial_state_estimator"]
        )

        config_path = "/home/adittrich/test_workspace/workspace/src/ball_models/config/config.toml"
        self.ball_model = ball_models.BallTrajectory(config_path)

        super().__init__(config=config, ball_model=self.ball_model)

        # Storage
        self.t_current = 0.0

        self.t = []
        self.z = []

        self.t_pred = []
        self.q_pred = []

        self.t_ests = []
        self.q_preds = []
        self.q_ests = []

    def initialize_predictor(
        self,
        time_stamp: float,
        q_init: Sequence[float],
        P_init: Optional[ndarray] = None,
    ) -> None:
        if P_init is None:
            P_init = eye(len(q_init))

        super().initialize_kalman(q_init, P_init)
        self.t_current = time_stamp

    def predict_horizon(self):
        self.reset_predictions()
        q = self.q

        t_predict = arange(self.t_current, self.t_prediction_horizon, dt)
        self.t_pred = t_predict

        duration = self.t_prediction_horizon - self.t_current
        dt = 1 / self.f_predictor

        self.q_pred = self.ball_model.simulate(q, duration, dt)

    def kalman_update_step(self, z: Sequence[float], time_stamp: float):
        dt = time_stamp - self.t_current
        q_pred, q_est, P_pred, P_est = self.predict_update(z, dt)

        self.t_current = time_stamp

        self.t_ests.append(time_stamp)
        self.q_preds.append(q_pred)
        self.q_ests.append(q_est)

    def get_prediction(
        self, filter=True
    ) -> Union[Sequence[float], Sequence[float], Sequence[float]]:
        t_pred = self.t_pred
        q_pred = self.q_pred

        if len(q_pred) == 0:
            return [], []

        if filter:
            t_pred, q_pred = self.prediction_filter.filter(t_pred, q_pred)

        return t_pred, q_pred

    def reset_predictions(self) -> None:
        self.t_pred = []
        self.q_pred = []
        self.P_pred = []

    def reset(self):
        self.t = []
        self.z = []

    def input_samples(self, z: Sequence[float], time_stamp: float):
        self.t.append(time_stamp)
        self.z.append(z)

        if self.init_buffer_size < len(self.z):
            self.kalman_update_step(z=z, time_stamp=time_stamp)
            self.predict_horizont()

        if self.init_buffer_size == len(self.z):
            q_init = self.inital_state_estimator(self.t, self.z)
            self.initialize_predictor(time_stamp=time_stamp, q_init=q_init)
            R = diag(std(self.z, axis=0))
            self.set_R(R)

        self.t_current = time_stamp

    def profile_predict(self):
        profile.runctx("self.predict_horizont()", globals(), locals())

    def profile_update(self, z, t):
        profile.runctx("self.kalman_update_step(z, t)", globals(), locals())
