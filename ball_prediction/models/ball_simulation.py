from typing import Callable, Sequence

import numpy as np


class BallSimulationSpin:
    def __init__(
        self,
        physics_config: dict,
        table_rebound_model: Callable,
        racket_rebound_model: Callable,
    ) -> None:
        self.m_ball = physics_config["ball_mass"]
        self.r_ball = physics_config["ball_radius"]
        self.rho = physics_config["air_density"]
        self.g = physics_config["gravitational_constant"]
        self.c_drag = physics_config["drag_coefficient"]
        self.c_lift = physics_config["lift_coefficient"]
        self.c_decay = physics_config["decay_coefficient"]

        self.A = np.pi * self.r_ball**2

        self.table_rebound_model = table_rebound_model
        self.racket_rebound_model = racket_rebound_model

    def check_table_contact(_q) -> bool:
        ...

    def check_racket_contact(_q) -> bool:
        ...

    def simulate(
        self, q_init, dt: float = 0.001, duration: float = 1.5, t_start: float = 0.0
    ):
        q = np.array(q_init)

        t = t_start
        t_end = t_start + duration

        q_history = [q]

        while t < t_end:
            _q = self.step(q=q, dt=dt)

            if self.check_table_contact(_q):
                _q = self.table_rebound_model(_q)

            if self.check_racket_contact(_q):
                _q = self.racket_rebound_model(_q)

            q = _q
            t += dt

            q_history.append(q)

        return q_history

    def step(self, q: Sequence[float], dt: float = 0.001):
        # Load variables for readability
        m_ball = self.m_ball
        r_ball = self.r_ball
        rho = self.rho
        g = self.g

        c_drag = self.c_drag
        c_lift = self.c_lift
        c_decay = self.c_decay

        A = self.A

        # Calculate new state
        q = np.array(q)

        k_magnus = 0.5 * rho * c_lift * A * r_ball / m_ball
        k_drag = -0.5 * rho * c_drag * A / m_ball
        k_gravity = g

        # Step calculation
        v = q[3:6]
        omega = q[6:9]

        F_gravity = k_gravity * np.array([0, 0, -1])
        F_drag = k_drag * np.norm(v) * v
        F_magnus = k_magnus * np.cross(omega, v)

        # System dynamics
        dv_dt = F_gravity + F_drag + F_magnus

        domega_dt = np.zeros(3)
        dq_dt = np.hstack((v, dv_dt, domega_dt))

        return q + dt * dq_dt


class BallSimulationMagnusForce:
    def __init__(
        self,
        physics_config: dict,
        table_rebound_model: Callable,
        racket_rebound_model: Callable,
    ) -> None:
        self.m_ball = physics_config["ball_mass"]
        self.r_ball = physics_config["ball_radius"]
        self.rho = physics_config["air_density"]
        self.g = physics_config["gravitational_constant"]
        self.c_drag = physics_config["drag_coefficient"]
        self.c_lift = physics_config["lift_coefficient"]
        self.c_decay = physics_config["decay_coefficient"]

        self.A = np.pi * self.r_ball**2

        self.table_rebound_model = table_rebound_model
        self.racket_rebound_model = racket_rebound_model

    def check_table_contact(_q) -> bool:
        ...

    def check_racket_contact(_q) -> bool:
        ...

    def simulate(
        self, q_init, dt: float = 0.001, duration: float = 1.5, t_start: float = 0.0
    ):
        q = np.array(q_init)

        t = t_start
        t_end = t_start + duration

        q_history = [q]

        while t < t_end:
            _q = self.step(q=q, dt=dt)

            if self.check_table_contact(_q):
                _q = self.table_rebound_model(_q)

            if self.check_racket_contact(_q):
                _q = self.racket_rebound_model(_q)

            q = _q
            t += dt

            q_history.append(q)

        return q_history

    def step(self, q: Sequence[float], dt: float = 0.001):
        # Load variables for readability
        m_ball = self.m_ball
        r_ball = self.r_ball
        rho = self.rho
        g = self.g

        c_drag = self.c_drag
        c_lift = self.c_lift
        c_decay = self.c_decay

        A = self.A

        # Calculate new state
        q = np.array(q)

        k_magnus = 0.5 * rho * c_lift * A * r_ball / m_ball
        k_drag = -0.5 * rho * c_drag * A / m_ball
        k_gravity = g

        # Step calculation
        v = q[3:6]
        F_magnus = q[6:9]

        F_gravity = k_gravity * np.array([0, 0, -1])
        F_drag = k_drag * np.norm(v) * v

        # System dynamics
        dv_dt = F_gravity + F_drag + F_magnus

        dF_magnus_dt = np.zeros(3)
        dq_dt = np.hstack((v, dv_dt, dF_magnus_dt))

        return q + dt * dq_dt
