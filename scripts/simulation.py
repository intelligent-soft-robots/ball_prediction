import time

import ball_models
import matplotlib.pyplot as plt
import numpy as np

model = ball_models.BallTrajectory(
    "/home/adittrich/test_workspace/workspace/src/ball_models/config/config.toml"
)

q = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0])

start_time = time.time()
trajectory = model.simulate(q, 1.0, 0.0005)
dt = time.time() - start_time
print(f"Runtime: {dt} s")

plt.plot(np.array(trajectory)[:, 2])
plt.show()
