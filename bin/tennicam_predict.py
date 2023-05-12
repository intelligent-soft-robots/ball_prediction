import pathlib
import tomlkit

import signal_handler
import tennicam_client
from numpy import hstack

from ball_prediction.trajectory_prediction import TrajectoryPredictor

TENNICAM_CLIENT_DEFAULT_SEGMENT_ID = "tennicam_client"


class TennicamClientPredictor:
    def __init__(self, config_path) -> None:
        # Tennicam initialisation
        global TENNICAM_CLIENT_DEFAULT_SEGMENT_ID
        self.frontend = tennicam_client.FrontEnd(TENNICAM_CLIENT_DEFAULT_SEGMENT_ID)

        # Predictor initialisation
        self.n_negative_ball_id = 0
        self.predictor = TrajectoryPredictor(load_toml(config_path))

    def run_predictor(self):
        iteration = self.frontend.latest().get_iteration()
        try:
            while not signal_handler.has_received_sigint():
                iteration += 1
                obs = self.frontend.read(iteration)

                ball_id = obs.get_ball_id()

                if ball_id != -1:
                    time_stamp = obs.get_time_stamp() * 10e-9
                    position = obs.get_position()
                    velocity = obs.get_velocity()
                    z = hstack((position, velocity))

                    self.predictor.input_samples(z, time_stamp)
                    self.predictor.predict_horizon()
                    prediction = self.predictor.get_prediction()

                    print(f"Position: {position}, Predict: {prediction}")

                    if prediction[0]:
                        raise KeyboardInterrupt(f"interrupt")

                if ball_id == -1:
                    self.n_negative_ball_id += 1

                    if self.n_negative_ball_id == 20:
                        self.n_negative_ball_id = 0
                        self.predictor.reset_predictions()

        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception as e:
            print("Error:", e)


def load_toml(file_path: str):
    with open(pathlib.Path(file_path), mode="r") as fp:
        config = fp.read()
        config_dict = dict(tomlkit.parse(config))

    return config_dict


def run_predictor():
    path = "/home/adittrich/test_workspace/workspace/src/ball_prediction/config/config.toml"
    predictor = TennicamClientPredictor(path)

    predictor.run_predictor()


if __name__ == "__main__":
    run_predictor()
