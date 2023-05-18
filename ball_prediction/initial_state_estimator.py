import logging
from typing import Sequence, Union

from numpy import array, gradient, hstack, ndarray
from numpy.polynomial.polynomial import Polynomial


class DirectMeasurement:
    """Directly extracts initial state from given measurements."""

    def estimate(
        self, time_stamps: Sequence[float], measurements: Sequence[Sequence[float]]
    ) -> Sequence[float]:
        """Estimates initial state by returning the last sample for measurements.

        Args:
            time_stamps (Sequence[float]): Time stamps. Unused for this estimator.
            measurements (Sequence[float]): Measurements with same dimension as the
            required state vector.

        Returns:
            Sequence[float]: Initial state
        """
        measurements = array(measurements)

        return measurements[-1]

    def estimate_extended_state_space(
        self, time_stamps: Sequence[float], measurements: Sequence[Sequence[float]]
    ) -> Sequence[float]:
        """Extends estimate with Zero vector.

        Args:
            time_stamps (Sequence[float]): Time_stamps. Unused for this estimator.
            measurements (Sequence[Sequence[float]]): Measurements with same dimension
            as the required state excluding the Zero tripple vector.

        Returns:
            Sequence[float]: Initital state.
        """
        estimate = self.estimate(time_stamps=time_stamps, measurements=measurements)

        logging.info("\033[31m" + f"{estimate}" + "\033[0m")

        return hstack((estimate, [0, 0, 0]))


class FiniteDifferencesEstimator:
    """Initial state is estimated by finite differences."""

    def estimate(
        self, time_stamps: Sequence[float], measurements: Sequence[Sequence[float]]
    ) -> Sequence[float]:
        """Estimates the initial state vector using finite differences.

        Args:
            time_stamps (Sequence[float]): Time stamps.
            measurements (Sequence[Sequence[float]]): Sequence of measurements.

        Raises:
            ValueError: Raises an error if not enough measurements are given.
            Approach requires at least two samples.

        Returns:
            ndarray: Initial state vector.
        """
        measurements = array(measurements)

        if measurements.size <= 1:
            raise ValueError(
                f"Number of measurements ({len(measurements)}) not sufficient"
            )

        velocities = []
        for axis in range(3):
            velocity = gradient(measurements[:, axis], time_stamps, axis=0)
            velocities.append(velocity[-1])

        velocities = array(velocities)

        return hstack((measurements[-1, 0:3], velocities))

    def estimate_extended_state_space(
        self, time_stamps: Sequence[float], measurements: Sequence[Sequence[float]]
    ) -> Sequence[float]:
        """Extends the estimated state vector with an zero vector.

        Args:
            time_stamps (Sequence[float]): Time stamps.
            measurements (Sequence[Sequence[float]]): Sequence of measurements.

        Returns:
            Sequence[float]: Initial state vector.
        """
        estimate = self.estimate(time_stamps=time_stamps, measurements=measurements)

        logging.info(estimate)

        return hstack((estimate, [0, 0, 0]))


class RegressionEstimator:
    """Initital state is measured by regression of the measurements."""

    def __init__(self) -> None:
        """Inititalizes estimator."""
        self.regression_degree = 3

    def load_config(self, config: dict) -> None:
        """Loads the required parameters given by configuration dict.

        Args:
            config (dict): Configuration dict.
        """
        self.regression_degree = config["initial_state_estimation"]["regression_degree"]

    def estimate(
        self, time_stamps: Sequence[float], measurements: Sequence[Sequence[float]]
    ) -> Sequence[float]:
        """Estimates the initital state by using regression.

        Args:
            time_stamps (Sequence[float]): Time stamps.
            measurements (Sequence[Sequence[float]]): Sequence of measurements.

        Returns:
            Sequence[float]: Initial state.
        """
        time_stamps = array(time_stamps)
        measurements = array(measurements)

        if len(measurements) < self.regression_degree:
            raise AttributeError(
                f"Not enough samples ({len(measurements)}) given for estimator method."
                f"Estimator with regression degree {self.regression_degree} requires at"
                f"least {self.regression_degree} measurements"
                "Consider raising the size of initial measurement buffer."
            )

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
        self, time_stamps: Sequence[float], measurements: Sequence[Sequence[float]]
    ) -> Sequence[float]:
        """Extends estimate with zero vector.

        Args:
            time_stamps (Sequence[float]): Time stamps.
            measurements (Sequence[Sequence[float]]): Sequence of measurements.

        Returns:
            Sequence[float]: Initital state.
        """
        estimate = self.estimate(time_stamps=time_stamps, measurements=measurements)
        logging.info(estimate)
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


class InitialStateEstimator:
    """Wrapper class for loading initial state estimators."""

    def load_estimator(
        config: dict,
    ) -> Union[
        DirectMeasurement,
        FiniteDifferencesEstimator,
        RegressionEstimator,
        BallLauncherModelState,
    ]:
        """Load an initial state estimator based on the provided configuration.

        Args:
            config (dict): A dictionary containing the configuration parameters.

        Raises:
            ValueError: Raised if estimator_name is invalid.

        Returns:
            InitialStateEstimator: An instance of the selected initial state estimator.
        """
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
