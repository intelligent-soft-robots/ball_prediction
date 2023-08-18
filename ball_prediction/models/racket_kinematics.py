import numpy as np

def compute_racket_orientation(
    joint_angles_rad,
):
    theta_1, theta_2, theta_3, theta_4 = joint_angles_rad

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

    # Assuming robot is set up not perfectly within tennicam coordinates
    theta_x_error = 0.0
    theta_y_error = 0.0
    theta_z_error = 0.0

    T_X = rotation_matrix_x(theta_x_error)
    T_Y = rotation_matrix_y(theta_y_error)
    T_Z = rotation_matrix_z(theta_z_error)

    T_COMP = T_Z @ T_Y @ T_X

    racket_normal = [0.0, 1.0, 0.0, 1.0]

    T_10 = rotation_matrix_z(theta_1)
    T_21 = rotation_matrix_y(theta_2)
    T_43 = rotation_matrix_y(theta_3)
    T_65 = rotation_matrix_z(theta_4)

    T_60 = T_65 @ T_43 @ T_21 @ T_10
    T_60 = T_10 @ T_21 @ T_43 @ T_65 
    
    trans_racket_normal = T_60 @ T_COMP @ racket_normal

    return trans_racket_normal[:3]

def compute_racket_velocity():
    pass