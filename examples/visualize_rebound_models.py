import matplotlib.pyplot as plt
import numpy as np

from ball_prediction.models.racket_rebound_model import *
from ball_prediction.models.table_rebound_model import *
from ball_prediction.models.utils import ContactType
from ball_prediction.utils.data_management import load_ball_data, load_robot_ball_data

np.set_printoptions(suppress=True)

from ball_prediction.models.ball_simulation import BallSimulationSpin
from ball_prediction.models.rebound_detection import detect_rebounds

file_path_nospin = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/no_spin_robot_simple.hdf5"
file_path_backspin = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/mn5008_backspin.hdf5"
file_path_topspin = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/mn5008_topspin.hdf5"
file_path_robot = "/home/lis/workspace/spin_project/workspace/src/ball_prediction/data/no_spin_robot.hdf5"

collection_nospin = load_ball_data(file_path_nospin)
collection_backspin = load_ball_data(file_path_backspin)
collection_topspin = load_ball_data(file_path_topspin)
collection_robot = load_robot_ball_data(file_path_robot)

collection = collection_robot

index = 14
start = 10
end = 50

ball_time_stamps = collection[index]["ball_time_stamps"]
ball_positions = collection[index]["ball_positions"]
ball_velocities = collection[index]["ball_velocities"]

time_stamps = np.array(ball_time_stamps)
positions = np.array(ball_positions)
velocities = np.array(ball_velocities)

omega_nospin = [0.0, 0.0, 0.0]
omega_backspin = [0.0, -60.0, 0.0]
omega_topspin = [0.0, 0.0, 0.0]

PHYSICS_CFG = {
    "ball_mass": 0.027,
    "ball_radius": 0.02,
    "air_density": 1.18,
    "gravitational_constant": 9.80801,
    "drag_coefficient": 0.47016899,
    "lift_coefficient": 1.46968343,
    "decay_coefficient": 0.005054473513775579,
}

sim = BallSimulationSpin(PHYSICS_CFG, None, None)
contact_dict = detect_rebounds(time_stamps, positions)

dt = 0.01
duration = 0.3

tennicam_dt = 0.005
steps_after_rebound = int(duration / tennicam_dt)

table_contacts = 0

for key, item in contact_dict.items():
    if item == ContactType.TABLE:
        table_contacts += 1

TRM_models = []

TRM_models.append(SimpleTableContact)
TRM_models.append(Hayakawa2016TableContact)
TRM_models.append(Zhang2014TableContact)
TRM_models.append(Nakashima2010TableContact)
TRM_models.append(Huang2011TableContact)
TRM_models.append(Zhang2010TableContact)


colors = [
    "#FF0000",
    "#00A79F",
    "#007480",
    "#413D3A",
    "#CAC7C7",
    "#B51F1F",
]  # 10 colors

TRM_models_label = [
    "Simple",
    "Hayakava et al. 2021",
    "Zhang et al. 2014",
    "Nakashima et al. 2010",
    "Huang et al. 2011",
    "Zhang et al. 2010",
]

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    ball_positions[:, 0],
    ball_positions[:, 1],
    ball_positions[:, 2],
)

num_plotted = 1
for i, (key, value) in enumerate(contact_dict.items(), start=1):
    if value == ContactType.TABLE:
        index = key

        # Rebound model
        v = np.hstack((ball_velocities[index - 3], [0, 0, 0]))

        iter = 0
        for label, model_gen in zip(TRM_models_label, TRM_models):
            model = model_gen()
            _v = model.forward(v)
            q = np.hstack((ball_positions[index], _v))

            # Simulate table tennis
            q_history = sim.simulate(q, dt, duration)
            q_history = np.array(q_history)
            ax.plot(
                q_history[:, 0],
                q_history[:, 1],
                q_history[:, 2],
                label=label,
                c=colors[iter],
            )
            iter += 1

        num_plotted += 1


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend()

ax.set_aspect("equal")

plt.show()
