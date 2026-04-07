'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    Warps an angle into [-pi, pi]. Used in the update step.
    :param angle_rad: Input angle in radians.
    :return: Angle wrapped to [-pi, pi].
    """
    # Normalize by subtracting integer multiples of 2*pi
    n = np.floor((angle_rad + np.pi) / (2 * np.pi))
    return angle_rad - 2 * np.pi * n


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    Initialize landmarks from initial pose and (beta, r) measurements and their covariances.
    :param init_measure: (beta0, r0, beta1, r1, ...).
    :param init_measure_cov: 2x2 covariance per landmark.
    :param init_pose: 3x1 initial pose.
    :param init_pose_cov: 3x3 pose covariance.
    :return: k, landmark (2k x 1), landmark_cov (2k x 2k).
    '''
    k = init_measure.shape[0] // 2
    x = float(init_pose[0, 0])
    y = float(init_pose[1, 0])
    theta = float(init_pose[2, 0])

    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))

    for i in range(k):
        beta = float(init_measure[2 * i, 0])
        r = float(init_measure[2 * i + 1, 0])
        # Landmark in world frame: lx = x + r*cos(theta+beta), ly = y + r*sin(theta+beta)
        c_tb = np.cos(theta + beta)
        s_tb = np.sin(theta + beta)
        landmark[2 * i] = x + r * c_tb
        landmark[2 * i + 1] = y + r * s_tb

        # Jacobian of (lx, ly) w.r.t. (x, y, theta, beta, r); then C_out = J @ C_in @ J.T
        dtheta = -r * s_tb
        dbeta = -r * s_tb
        J_pose = np.array([[1, 0, dtheta], [0, 1, r * c_tb]])
        J_meas = np.array([[dbeta, c_tb], [r * c_tb, s_tb]])
        J = np.hstack([J_pose, J_meas])
        C_in = np.block([[init_pose_cov, np.zeros((3, 2))], [np.zeros((2, 3)), init_measure_cov]])
        block = J @ C_in @ J.T
        landmark_cov[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    Predict step in EKF SLAM: motion model and covariance propagation.
    :param X: State (3+2k x 1). :param P: Covariance (3+2k x 3+2k).
    :param control: (d, alpha) polar control. :param control_cov: 3x3 in (x,y,theta).
    :param k: Number of landmarks.
    :return: X_pre, P_pre.
    '''
    x, y, theta = X[0, 0], X[1, 0], X[2, 0]
    d = float(control[0, 0])
    alpha = float(control[1, 0])

    # Only robot pose changes: pose_new = pose + [d*cos(theta), d*sin(theta), alpha]
    motion = np.array([[d * np.cos(theta)], [d * np.sin(theta)], [alpha]])
    F = np.vstack([np.eye(3), np.zeros((2 * k, 3))])
    X_pre = X + F @ motion

    # Jacobian of new pose w.r.t. (x, y, theta)
    J_pose = np.array([
        [1, 0, -d * np.sin(theta)],
        [0, 1, d * np.cos(theta)],
        [0, 0, 1]
    ], dtype=float)
    G = F @ J_pose @ F.T
    np.fill_diagonal(G, 1.0)

    # Rotate control noise from robot frame to world frame
    c, s = np.cos(theta), np.sin(theta)
    H = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    Q = F @ H @ control_cov @ H.T @ F.T
    P_pre = G @ P @ G.T + Q

    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    Update step in EKF SLAM: correct state with measurements using Kalman gain.
    :param X_pre: Predicted state (3+2k x 1). :param P_pre: Predicted covariance.
    :param measure: Measurements (2k x 1). :param measure_cov: 2x2 per landmark.
    :param k: Number of landmarks.
    :return: Updated X, P.
    '''
    x = X_pre[0, 0]
    y = X_pre[1, 0]
    theta = X_pre[2, 0]
    n = 3 + 2 * k

    for i in range(k):
        x = X_pre[0, 0]
        y = X_pre[1, 0]
        theta = X_pre[2, 0]
        lx = X_pre[3 + 2 * i, 0]
        ly = X_pre[3 + 2 * i + 1, 0]
        dx = lx - x
        dy = ly - y
        q = dx * dx + dy * dy
        q = max(q, 1e-6)  # avoid singular H when landmark is very close

        # Predicted measurement: bearing and range
        h_bearing = warp2pi(np.arctan2(dy, dx) - theta)
        h_range = np.sqrt(q)
        h = np.array([[h_bearing], [h_range]])

        # Jacobian of h w.r.t. (x, y, theta, lx, ly); then map to full state via F
        F_block = np.zeros((5, n))
        F_block[:3, :3] = np.eye(3)
        F_block[3:, 3 + 2 * i : 3 + 2 * i + 2] = np.eye(2)
        sq = np.sqrt(q)
        H_local = (1.0 / q) * np.array([
            [dy, -dx, -q, -dy, dx],
            [-sq * dx, -sq * dy, 0, sq * dx, sq * dy]
        ])
        H = (H_local @ F_block).astype(float)

        S = H @ P_pre @ H.T + measure_cov
        K = P_pre @ H.T @ np.linalg.inv(S)
        # Wrap bearing residual to [-pi, pi] so EKF corrects in the right direction
        y_res = np.array([[warp2pi(float(measure[2 * i, 0]) - float(h[0, 0]))], [float(measure[2 * i + 1, 0]) - float(h[1, 0])]])
        X_pre = X_pre + K @ y_res
        # Joseph form to keep P positive definite (more stable than I-KH)
        I_KH = np.eye(n) - K @ H
        P_pre = I_KH @ P_pre @ I_KH.T + K @ measure_cov @ K.T

    # Keep orientation in [-pi, pi] for consistent motion prediction
    X_pre[2, 0] = warp2pi(X_pre[2, 0])
    return X_pre, P_pre


def evaluate(X, P, k):
    '''
    Evaluate EKF SLAM: plot ground truth landmarks and report Euclidean/Mahalanobis distances.
    :param X: State (3+2k x 1). :param P: Covariance. :param k: Number of landmarks.
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2], c='r')
    plt.draw()
    plt.waitforbuttonpress(0)

    l_pred = np.squeeze(X[3:])
    for i in range(k):
        err = l_true[2 * i : 2 * i + 2] - l_pred[2 * i : 2 * i + 2]
        euclidean_dist = np.linalg.norm(err)
        Pi = P[3 + 2 * i : 3 + 2 * i + 2, 3 + 2 * i : 3 + 2 * i + 2]
        mahalanobis_dist = np.sqrt(err @ np.linalg.inv(Pi) @ err)
        print("Landmark {}/{}: Euclidean_dist: {}, Mahalanobis_dist: {}".format(
            i + 1, k, euclidean_dist, mahalanobis_dist))


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.01
    sig_r = 0.08


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
