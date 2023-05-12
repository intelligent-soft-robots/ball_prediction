from typing import Sequence

from numpy import array, gradient, hstack
from numpy.polynomial.polynomial import Polynomial


class InitialStateEstimator:
    def load_estimator(config: dict):
        estimator_name = config["estimation_method"]
        estimators = {
            "direct": DirectMeasurement,
            "finite": FiniteDifferencesEstimator,
            "regression": RegressionEstimator,
        }
        if estimator_name in estimators:
            return estimators[estimator_name]()
        else:
            raise ValueError(f"Invalid class name: {estimator_name}")


class DirectMeasurement:
    def estimate(self, time_stamps: Sequence[float], measurements: Sequence[float]):
        measurements = array(measurements)

        return measurements[-1]

    def estimate_extended_state_space(
        self, time_stamps: Sequence[float], measurements: Sequence[float]
    ):
        estimate = self.estimate(time_stamps=time_stamps, measurements=measurements)
        print(estimate)
        return hstack((estimate, [0, 0, 0]))


class FiniteDifferencesEstimator:
    def estimate(self, time_stamps: Sequence[float], measurements: Sequence[float]):
        measurements = array(measurements)

        if measurements.size <= 1:
            raise ValueError(
                f"Number of measurements ({len(measurements)}) not sufficient"
            )

        velocities = []
        for axis in range(3):
            velocity = gradient(measurements[:, axis], time_stamps)
            velocities.append(velocity)

        velocities = array(velocities)

        return hstack((measurements[-1], velocities[-1]))

    def estimate_extended_state_space(
        self, time_stamps: Sequence[float], measurements: Sequence[float]
    ):
        estimate = self.estimate(time_stamps=time_stamps, measurements=measurements)
        return hstack((estimate, [0, 0, 0]))


class RegressionEstimator:
    def __init__(self) -> None:
        self.regression_degree = 3

    def load_config(self, config: dict) -> None:
        self.regression_degree = config["initial_state_estimation"]["regression_degree"]

    def estimate(self, time_stamps: Sequence[float], measurements: Sequence[float]):
        time_stamps = array(time_stamps)
        measurements = array(measurements)

        position = []
        velocity = []

        for axis in range(3):
            position_polynomial = Polynomial.fit(
                time_stamps[:-1], measurements[:-1, axis], deg=self.regression_degree
            )

            velocity_polynomial = position_polynomial.deriv()

            position.append(position_polynomial(time_stamps[-1]))
            velocity.append(velocity_polynomial(time_stamps[-1]))

        return hstack((position, velocity))

    def estimate_extended_state_space(
        self, time_stamps: Sequence[float], measurements: Sequence[float]
    ):
        estimate = self.estimate(time_stamps=time_stamps, measurements=measurements)
        return hstack((estimate, [0, 0, 0]))


class BallLauncherModelState:
    def __init__(self) -> None:
        pass

    def load_config(self, config: dict) -> None:
        pass

    def estimate(
        self,
        time_stamps: Sequence[float],
        measurements: Sequence[float],
        launch_parameters: Sequence[float],
    ):
        pass
