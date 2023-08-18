import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from ball_prediction.ball_prediction.models.racket_kinematics import (
    compute_racket_orientation,
)


def test_orientation_calculation():
    np.set_printoptions(suppress=True)
    pi = math.pi

    joint_angles_rad = [40, 00, 0.0, 0.0]
    length_upper_arm = 0.5
    lenght_lower_arm = 0.4
    degrees = True

    theta_1, theta_2, theta_3, theta_4 = joint_angles_rad

    o = [0.0, 0.0, 0.0]

    o_0 = [0.0, 1.0, 0.0]

    R1 = Rotation.from_rotvec(np.deg2rad([0.0, 0.0, theta_1]))  # Shoulder joint
    R2 = Rotation.from_rotvec(np.deg2rad([0.0, theta_2, 0.0]))  # Shoulder joint

    R12 = R2 * R1
    T1 = [0, 0, length_upper_arm]
    R3 = Rotation.from_rotvec(np.deg2rad([0.0, theta_3, 0.0]))  # Elbow joint
    T2 = [0, 0, lenght_lower_arm]
    R4 = Rotation.from_rotvec(np.deg2rad([0.0, 0.0, theta_4]))  # Wrist joint

    o_2 = R12.apply(o_0)
    o_2 += T1

    o_3 = R3.apply(o_2)
    o_3 += T2

    o_4 = R4.apply(o_3)

    # Visualisation
    ax = plt.figure().add_subplot(projection="3d")
    x_axis = np.array([0.4, 0, 0])
    y_axis = np.array([0, 0.4, 0])
    z_axis = np.array([0, 0, 0.4])

    # Draw the basis vectors as arrows
    ax.quiver(*o, *x_axis, color="r", label="X-axis")
    ax.quiver(*o, *y_axis, color="g", label="Y-axis")
    ax.quiver(*o, *z_axis, color="b", label="Z-axis")

    ax.scatter(*o, s=20, c="r", alpha=0.5)

    # Shoulder
    P0 = o
    print(f"P0: {P0}")
    ax.scatter(*P0, s=20, c="b", alpha=0.5)

    # Elbow
    P1 = o_2
    print(f"P1: {P1}")
    ax.scatter(*P1, s=20, c="g", alpha=0.5)

    # Wrist
    P2 = o_3
    print(f"P2: {P2}")
    ax.scatter(*P2, s=20, c="black", alpha=0.5)

    # Shoulder to Elbow
    P0P1 = [[i, j] for i, j in zip(P0, P1)]
    print(P0P1)
    ax.plot(*P0P1)

    # Elbow to Wrist
    P1P2 = [[i, j] for i, j in zip(P1, P2)]
    print(P1P2)
    ax.plot(*P1P2)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pass


def test_orientation_calculation_homogenous():
    import math

    import numpy as np
    from scipy.spatial.transform import Rotation

    from ball_prediction.ball_prediction.models.racket_kinematics import (
        compute_racket_orientation,
    )

    np.set_printoptions(suppress=True)
    pi = math.pi

    joint_angles_deg = [20, 30, 00, 0]
    joint_angles_rad = np.deg2rad(joint_angles_deg)
    length_upper_arm = 0.5
    length_lower_arm = 0.4

    theta_1, theta_2, theta_3, theta_4 = joint_angles_rad

    o = [0.0, 0.0, 0.0]

    def translation_matrix(x, y, z):
        return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

    def rotation_matrix_x(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    def rotation_matrix_y(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    def rotation_matrix_z(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    print(theta_1)
    print(theta_2)
    print(theta_3)
    print(theta_4)

    T_10 = rotation_matrix_z(theta_1)
    T_21 = rotation_matrix_y(theta_2)
    T_32 = translation_matrix(0, 0, length_upper_arm)

    T_43 = rotation_matrix_y(theta_3)
    T_54 = translation_matrix(0, 0, length_lower_arm)
    T_65 = rotation_matrix_z(theta_4)

    origin = np.array([0, 0, 0])

    homogeneous_coordinate = np.append(origin, 1)

    T_20 = T_21 @ T_10
    T_30 = T_32 @ T_21 @ T_10
    T_40 = T_43 @ T_32 @ T_21 @ T_10
    T_50 = T_54 @ T_43 @ T_32 @ T_21 @ T_10
    T_60 = T_65 @ T_54 @ T_43 @ T_32 @ T_21 @ T_10

    end_point_00 = homogeneous_coordinate  # origin
    end_point_10 = T_10 @ end_point_00
    end_point_20 = T_21 @ end_point_10
    end_point_30 = T_32 @ end_point_20  # joint 1
    end_point_40 = T_43 @ end_point_30
    end_point_50 = T_54 @ end_point_40  # joint 2
    end_point_60 = T_65 @ end_point_50

    # Plotting the 3D visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the joints
    joints = [end_point_00[:3], end_point_30[:3], end_point_50[:3]]
    for i, joint in enumerate(joints):
        ax.scatter(joint[0], joint[1], joint[2], c="r", marker="o", s=36)
        ax.text(
            joint[0],
            joint[1],
            joint[2],
            f"Joint {i}",
            fontsize=12,
            ha="right",
            va="bottom",
        )

    # Set axis labels (optional)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set the aspect ratio of the 3D plot to be equal
    ax.set_aspect("equal")

    # Add a legend for the coordinate system axes
    ax.legend()
