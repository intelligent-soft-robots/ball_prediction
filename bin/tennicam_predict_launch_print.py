"""
Launches table tennis ball and passes launch parameters to 
ball predictor for initial state estimation.
"""
import pathlib
import typing

import signal_handler
import tennicam_client
import tomlkit
from ball_launcher_beepy.ball_launcher_client import BallLauncherClient
from numpy import hstack

from ball_prediction.trajectory_prediction import TrajectoryPredictor

IP = "10.42.26.171"  # AIMY IP
PORT = 5555


class _BallLauncherConfig:
    """Stores launch configuration of launch dialog."""

    def __init__(self) -> None:
        # reasonable default values
        self.ip = IP
        self.port = PORT
        self.phi = 0.5
        self.theta = 0.5
        self.top_left_motor = 0.5
        self.top_right_motor = 0.5
        self.bottom_motor = 0.5


def _dialog() -> _BallLauncherConfig:
    """Configuration dialog, provides reasonable default values."""
    config = _BallLauncherConfig()

    args = (
        ("ip", str),
        ("port", int),
        ("phi", float),
        ("theta", float),
        ("top_left_motor", float),
        ("top_right_motor", float),
        ("bottom_motor", float),
    )

    def _get_user_input(
        arg: str, type_: typing.Union[str, int, float], config: _BallLauncherConfig
    ) -> bool:
        # returns None if keyboard interrupt, user entered value otherwise
        ok = False
        while not ok:
            value = input(
                str("\tvalue for {} ({}): ").format(arg, getattr(config, arg))
            )
            if value == "":
                # user pressed enter, using default value
                value = getattr(config, arg)
            try:
                value = type_(value)
                ok = True
            except ValueError:
                print("\t\terror, could not cast to", type_)
            except KeyboardInterrupt:
                return None
        return value

    for arg, type_ in args:
        value = _get_user_input(arg, type_, config)
        if value is None:
            return None
        else:
            setattr(config, arg, value)

    return config


def _launch(config: _BallLauncherConfig) -> None:
    """
    Launches the ball according to the provided
    configuration
    """

    client = BallLauncherClient(config.ip, config.port)
    client.set_rpm(
        config.phi,
        config.theta,
        config.top_left_motor,
        config.top_right_motor,
        config.bottom_motor,
    )

    client.launch_ball()


def _launching_dialog():
    path = "/home/adittrich/test_workspace/workspace/src/ball_prediction/config/config.toml"
    predictor = TennicamClientPredictor(path)

    print()
    config = _dialog()
    predictor.run_predictor()
    if config is not None:
        _launch(config)

    print()


class TennicamClientPredictor:
    def __init__(self, config_path) -> None:
        # Tennicam initialisation
        global TENNICAM_CLIENT_DEFAULT_SEGMENT_ID
        self.frontend = tennicam_client.FrontEnd(TENNICAM_CLIENT_DEFAULT_SEGMENT_ID)

        config = load_toml(config_path)
        self.negative_ball_threshold = config["misc"]["n_negative_ball_threshold"]

        # Predictor initialisation
        self.n_negative_ball_id = 0
        self.predictor = TrajectoryPredictor(config)

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

                    print(f"Position: {position}, Prediction: {prediction}")

                    if prediction[0]:
                        raise KeyboardInterrupt(f"interrupt")

                if ball_id == -1:
                    self.n_negative_ball_id += 1

                    if self.n_negative_ball_id == self.negative_ball_threshold:
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


if __name__ == "__main__":
    _launching_dialog()
