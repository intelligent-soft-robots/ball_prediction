import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Sample data: Replace this with your 12000 ball trajectories.
# Each trajectory is represented as a list of (x, y, z) coordinates.
ball_trajectories = [np.random.rand(50, 3) for _ in range(12000)]

class BallTrajectoryViewer:
    def __init__(self, root, trajectories):
        self.root = root
        self.trajectories = trajectories
        self.filtered_trajectories = []

        self.figure = plt.figure()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.remove_buttons_top = []
        self.remove_buttons_bottom = []

        self.show_trajectories()  # Initial display of trajectories

    def show_trajectories(self):
        self.figure.clear()
        self.remove_buttons_top = []
        self.remove_buttons_bottom = []

        # Create a frame for the top row of buttons
        top_button_frame = ttk.Frame(self.root)
        top_button_frame.pack(side="top")

        # Create a frame for the bottom row of buttons
        bottom_button_frame = ttk.Frame(self.root)
        bottom_button_frame.pack(side="bottom")

        for i in range(5):
            if not self.trajectories:
                break
            trajectory = self.trajectories.pop(0)  # Pop from the front

            ax = self.figure.add_subplot(2, 5, i + 1, projection='3d')
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
            ax.set_title(f"Trajectory 1")

            label = tk.Label(top_button_frame, text=f"Trajectory 1")
            label.grid(row=0, column=i)

            remove_button = ttk.Button(top_button_frame, text="Remove", command=lambda traj=trajectory: self.remove_trajectory(traj))
            remove_button.grid(row=1, column=i)

            self.remove_buttons_top.append((remove_button, label))

        for i in range(5):
            if not self.trajectories:
                break
            trajectory = self.trajectories.pop(0)  # Pop from the front

            ax = self.figure.add_subplot(2, 5, i + 6, projection='3d')  # Start from the 6th subplot
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
            ax.set_title(f"Trajectory 1")

            label = tk.Label(bottom_button_frame, text=f"Trajectory 1")
            label.grid(row=0, column=i)

            remove_button = ttk.Button(bottom_button_frame, text="Remove", command=lambda traj=trajectory: self.remove_trajectory(traj))
            remove_button.grid(row=1, column=i)

            self.remove_buttons_bottom.append((remove_button, label))

        self.canvas.draw()

    def remove_trajectory(self, trajectory_to_remove):
        self.trajectories = [traj.tolist() for traj in self.trajectories]
        if tuple(trajectory_to_remove) in map(tuple, self.trajectories):
            self.trajectories = [traj for traj in self.trajectories if tuple(traj) != tuple(trajectory_to_remove)]
        self.show_trajectories()
        if not self.trajectories:
            for button, _ in self.remove_buttons_top + self.remove_buttons_bottom:
                button.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Ball Trajectory Viewer")
    app = BallTrajectoryViewer(root, ball_trajectories.copy())  # Make a copy to avoid modifying the original data
    root.mainloop()
