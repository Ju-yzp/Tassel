"""
Multi-landmark time-delay Jacobian verification.

Uses N landmarks spread across the image at different depths to create
a well-conditioned, geometrically anisotropic test. Compares the stacked
analytical Jacobian against stacked finite-difference.
"""

import numpy as np
from numpy.linalg import norm


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(phi):
    th = norm(phi)
    if th < 1e-12:
        return np.eye(3)
    a = phi / th
    return np.eye(3) + np.sin(th) * skew(a) + (1 - np.cos(th)) * skew(a) @ skew(a)


def imu_at(imu_ts, R_gt, v_gt, P_gt, gyro_body, acc_body, t):
    idx = np.searchsorted(imu_ts, t)
    if idx <= 0:
        return (R_gt[0].copy(), v_gt[0].copy(), P_gt[0].copy(),
                gyro_body[0].copy(), acc_body[0].copy())
    if idx >= len(imu_ts):
        return (R_gt[-1].copy(), v_gt[-1].copy(), P_gt[-1].copy(),
                gyro_body[-1].copy(), acc_body[-1].copy())
    i0, i1 = idx - 1, idx
    dt = t - imu_ts[i0]
    if abs(dt) < 1e-12:
        return (R_gt[i0].copy(), v_gt[i0].copy(), P_gt[i0].copy(),
                gyro_body[i0].copy(), acc_body[i0].copy())
    alpha = dt / (imu_ts[i1] - imu_ts[i0])
    gyro = (1 - alpha) * gyro_body[i0] + alpha * gyro_body[i1]
    acc = (1 - alpha) * acc_body[i0] + alpha * acc_body[i1]
    gm = 0.5 * (gyro_body[i0] + gyro)
    am = 0.5 * (acc_body[i0] + acc)
    R = R_gt[i0] @ exp_so3(gm * dt)
    v = v_gt[i0] + R_gt[i0] @ am * dt
    P = P_gt[i0] + v_gt[i0] * dt + 0.5 * R_gt[i0] @ am * dt ** 2
    return R, v, P, gyro, acc


def generate_imu(duration, freq):
    dt = 1.0 / freq
    n = int(duration / dt)
    ts = np.arange(n) * dt
    gyro = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    for i in range(n):
        t = ts[i]
        gyro[i] = [2.0 * np.sin(2 * np.pi * 0.6 * t),
                   1.5 * np.cos(2 * np.pi * 0.9 * t + 1.0),
                   1.0 * np.sin(2 * np.pi * 1.3 * t + 0.3)]
        acc[i] = [3.0 * np.cos(2 * np.pi * 0.4 * t + 0.7),
                  2.0 * np.sin(2 * np.pi * 0.55 * t),
                  1.0 * np.cos(1.8 * np.pi * t) + 9.81]
    R = np.zeros((n, 3, 3)); R[0] = np.eye(3)
    v = np.zeros((n, 3)); P = np.zeros((n, 3))
    for i in range(n - 1):
        gm = 0.5 * (gyro[i] + gyro[i + 1])
        am = 0.5 * (acc[i] + acc[i + 1])
        R[i + 1] = R[i] @ exp_so3(gm * dt)
        v[i + 1] = v[i] + R[i] @ am * dt
        P[i + 1] = P[i] + v[i] * dt + 0.5 * R[i] @ am * dt ** 2
    return ts, R, v, P, gyro, acc


# ── Analytical Jacobians (single landmark → 2D vector) ──

def jac_user(pt_i_C, R_i, R_j, P_i, P_j, w_i, w_j, v_i, v_j,
             a_i, a_j, ric, tic, reduce, dt):
    """User's formula from visual_factor.cpp lines 83-90."""
    p_I_i = ric @ pt_i_C + tic
    inner = (
        skew(w_j) @ R_j.T @ (R_i @ p_I_i + P_i)
        - R_j.T @ (R_i @ skew(w_i) @ p_I_i + v_i
                   + (R_i @ exp_so3(w_i * dt) @ a_i + R_i @ a_i) * dt)
        - skew(w_j) @ R_j.T @ P_j
        + R_j.T @ (v_j + (R_j @ exp_so3(w_j * dt) @ a_j + R_j @ a_j) * dt)
    )
    return reduce @ ric.T * (-1.0) @ inner


def jac_corrected(pt_i_C, R_i, R_j, P_i, P_j, w_i, w_j, v_i, v_j,
                  a_i, a_j, ric, tic, reduce, dt):
    """Sign-fixed formula."""
    p_I_i = ric @ pt_i_C + tic
    return reduce @ ric.T @ (
        + skew(w_j) @ R_j.T @ (R_i @ p_I_i + P_i)
        - skew(w_j) @ R_j.T @ P_j
        - R_j.T @ R_i @ skew(w_i) @ p_I_i
        - R_j.T @ v_i
        - R_j.T @ (R_i @ exp_so3(w_i * dt) @ a_i + R_i @ a_i) * dt
        + R_j.T @ v_j
        + R_j.T @ (R_j @ exp_so3(w_j * dt) @ a_j + R_j @ a_j) * dt
    )


# ── Old formula candidates (w, a both body-frame) ──

def jac_old_exact(p_G, R, P, w_body, v, a_body, ric, reduce, dt):
    """Old formula exactly as in updated code (third term +)."""
    inner = (
        skew(w_body) @ R.T @ p_G
        - skew(w_body) @ R.T @ P
        + R.T @ (v + (R @ exp_so3(w_body * dt) @ a_body + R @ a_body) * dt)
    )
    return reduce @ ric.T * (-1.0) @ inner


def jac_old_no_minus1(p_G, R, P, w_body, v, a_body, ric, reduce, dt):
    """Old formula without outer -1."""
    inner = (
        skew(w_body) @ R.T @ p_G
        - skew(w_body) @ R.T @ P
        - R.T @ (v + (R @ exp_so3(w_body * dt) @ a_body + R @ a_body) * dt)
    )
    return reduce @ ric.T @ inner


def jac_old_flip_signs(p_G, R, P, w_body, v, a_body, ric, reduce, dt):
    """Old formula with ALL inner signs flipped, no outer -1."""
    inner = (
        - skew(w_body) @ R.T @ p_G
        + skew(w_body) @ R.T @ P
        + R.T @ (v + (R @ exp_so3(w_body * dt) @ a_body + R @ a_body) * dt)
    )
    return reduce @ ric.T @ inner


# ── Generate well-spread landmarks ──

def generate_spread_landmarks(n_lm, K, tc_i, tc_j, td_true,
                               imu_ts, R_gt, v_gt, P_gt, gyro, acc, ric, tic):
    """Generate n_lm points well-spread in host camera frustum.
    Returns list of (pt_i_C, pt_j_C) in camera frames at true poses."""

    R_i_true, _, P_i_true, _, _ = imu_at(imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_i - td_true)
    R_j_true, _, P_j_true, _, _ = imu_at(imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_j - td_true)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    img_w, img_h = 640, 480

    # Divide image into a grid, sample points in each cell at varying depths
    grid = int(np.ceil(np.sqrt(n_lm)))
    landmarks = []

    for gx in range(grid):
        for gy in range(grid):
            if len(landmarks) >= n_lm:
                break
            # Jitter within grid cell
            u = (gx + 0.2 + 0.6 * np.random.random()) / grid * img_w
            v = (gy + 0.2 + 0.6 * np.random.random()) / grid * img_h
            # Varying depth: near to far
            depth = 2.0 + 8.0 * (gx * grid + gy) / (grid * grid) + np.random.uniform(-1, 1)

            # Point in host camera frame
            pt_C = np.array([(u - cx) * depth / fx,
                             (v - cy) * depth / fy,
                             depth])

            # Check visible in target
            p_G = R_i_true @ (ric @ pt_C + tic) + P_i_true
            p_I_j = R_j_true.T @ (p_G - P_j_true)
            p_C_j = ric.T @ (p_I_j - tic)
            if p_C_j[2] <= 0.1:
                continue
            uv_j = np.array([fx * p_C_j[0] / p_C_j[2] + cx,
                             fy * p_C_j[1] / p_C_j[2] + cy])
            if 15 <= uv_j[0] <= img_w - 15 and 15 <= uv_j[1] <= img_h - 15:
                landmarks.append((pt_C, p_C_j))

        if len(landmarks) >= n_lm:
            break

    return landmarks[:n_lm]


def main():
    np.random.seed(42)
    duration, imu_freq = 5.0, 400
    imu_ts, R_gt, v_gt, P_gt, gyro, acc = generate_imu(duration, imu_freq)

    ric = exp_so3(np.array([0.02, -0.01, 0.03]))
    tic = np.array([0.05, 0.01, -0.02])
    K = np.array([[400., 0, 320.], [0, 400., 240.], [0, 0, 1.]])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    dt_acc = 0.001
    eps = 1e-5
    n_lm = 25  # landmarks per pair

    print("=" * 100)
    print("Multi-landmark Jacobian verification")
    print(f"  Landmarks per test: {n_lm}")
    print(f"  Finite-diff step:  {eps}")
    print("=" * 100)

    for td_true in [0.0, 0.004, 0.015, 0.030]:
        print(f"\n─── t_d_true = {td_true * 1000:.0f} ms ───")
        n_tests = 0
        n_user_pass, n_corr_pass = 0, 0

        for _ in range(30):
            tc_i = np.random.uniform(0.5, duration - 1.5)
            tc_j = tc_i + np.random.uniform(0.1, 0.8)

            # Get landmarks
            lms = generate_spread_landmarks(
                n_lm, K, tc_i, tc_j, td_true,
                imu_ts, R_gt, v_gt, P_gt, gyro, acc, ric, tic)

            if len(lms) < 8:
                continue  # need enough landmarks
            n_tests += 1

            R_i, v_i, P_i, w_i_b, a_i_b = imu_at(
                imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_i - td_true)
            R_j, v_j, P_j, w_j_b, a_j_b = imu_at(
                imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_j - td_true)

            # Stack Jacobians and residuals for all landmarks
            n_pts = len(lms)
            J_user_all = np.zeros((2 * n_pts, 1))
            J_corr_all = np.zeros((2 * n_pts, 1))

            def residuals_stacked(td_shift):
                """Stacked residuals for all landmarks."""
                R_i_s, _, P_i_s, _, _ = imu_at(
                    imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_i - td_true + td_shift)
                R_j_s, _, P_j_s, _, _ = imu_at(
                    imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_j - td_true + td_shift)
                r_all = np.zeros(2 * n_pts)
                for k, (pt_C, p_C_j_meas) in enumerate(lms):
                    # Predicted p_C_j
                    p_G_s = R_i_s @ (ric @ pt_C + tic) + P_i_s
                    p_I_j_s = R_j_s.T @ (p_G_s - P_j_s)
                    p_C_j_s = ric.T @ (p_I_j_s - tic)
                    z_s = p_C_j_s[2]
                    uv_s = np.array([fx * p_C_j_s[0] / z_s + cx,
                                     fy * p_C_j_s[1] / z_s + cy])
                    # Measured (at true t_d=0 shift)
                    z_m = p_C_j_meas[2]
                    uv_m = np.array([fx * p_C_j_meas[0] / z_m + cx,
                                     fy * p_C_j_meas[1] / z_m + cy])
                    r_all[2 * k:2 * k + 2] = uv_m - uv_s
                return r_all

            # FD: ∂r/∂t_d.
            # td_shift > 0 → query later → effective t_d smaller.
            # r_plus  = r(td_true - eps), r_minus = r(td_true + eps)
            # ∂r/∂t_d ≈ (r(td_true + eps) - r(td_true - eps)) / (2eps)
            #          = (r_minus - r_plus) / (2eps)
            r_plus = residuals_stacked(+eps)   # query later  → t_d smaller
            r_minus = residuals_stacked(-eps)  # query earlier → t_d larger
            J_fd_all = (r_minus - r_plus) / (2 * eps)
            J_fd = J_fd_all.reshape(-1, 1)

            # Build analytical Jacobians for each landmark
            for k, (pt_C, p_C_j) in enumerate(lms):
                p_G = R_i @ (ric @ pt_C + tic) + P_i
                p_I_j = R_j.T @ (p_G - P_j)
                p_C_j_pred = ric.T @ (p_I_j - tic)
                z = p_C_j_pred[2]
                reduce = np.array([[fx / z, 0, -fx * p_C_j_pred[0] / (z * z)],
                                   [0, fy / z, -fy * p_C_j_pred[1] / (z * z)]])

                J_user_all[2 * k:2 * k + 2, :] = jac_user(
                    pt_C, R_i, R_j, P_i, P_j, w_i_b, w_j_b, v_i, v_j,
                    a_i_b, a_j_b, ric, tic, reduce, dt_acc).reshape(2, 1)

                J_corr_all[2 * k:2 * k + 2, :] = jac_corrected(
                    pt_C, R_i, R_j, P_i, P_j, w_i_b, w_j_b, v_i, v_j,
                    a_i_b, a_j_b, ric, tic, reduce, dt_acc).reshape(2, 1)

            # Cosine similarity between stacked Jacobians
            def cos_sim(a, b):
                return np.dot(a.ravel(), b.ravel()) / (norm(a) * norm(b) + 1e-15)

            cos_u = cos_sim(J_user_all, J_fd_all)
            cos_c = cos_sim(J_corr_all, J_fd_all)
            ang_u = np.degrees(np.arccos(np.clip(cos_u, -1, 1)))
            ang_c = np.degrees(np.arccos(np.clip(cos_c, -1, 1)))

            # Check Frobenius norm ratio
            ratio_u = norm(J_user_all) / (norm(J_fd_all) + 1e-15)
            ratio_c = norm(J_corr_all) / (norm(J_fd_all) + 1e-15)

            # Compare per-landmark direction consistency
            ang_u_per_lm = []
            ang_c_per_lm = []
            for k in range(n_pts):
                ju = J_user_all[2*k:2*k+2].ravel()
                jc = J_corr_all[2*k:2*k+2].ravel()
                jf = J_fd_all[2*k:2*k+2].ravel()
                cos_uk = np.dot(ju, jf) / (norm(ju) * norm(jf) + 1e-15)
                cos_ck = np.dot(jc, jf) / (norm(jc) * norm(jf) + 1e-15)
                ang_u_per_lm.append(np.degrees(np.arccos(np.clip(cos_uk, -1, 1))))
                ang_c_per_lm.append(np.degrees(np.arccos(np.clip(cos_ck, -1, 1))))

            ang_u_per = np.mean(ang_u_per_lm)
            ang_c_per = np.mean(ang_c_per_lm)

            status = "OK" if abs(cos_c) > 0.999 else "FAIL"
            if abs(cos_c) > 0.999:
                n_corr_pass += 1
            if abs(cos_u) > 0.999:
                n_user_pass += 1

            print(f"  [{n_tests:2d}] n_lm={n_pts:2d}  "
                  f"user={ang_u:6.1f}deg (per={ang_u_per:5.1f}deg)  "
                  f"corr={ang_c:6.1f}deg (per={ang_c_per:5.1f}deg)  "
                  f"ratio_u={ratio_u:.3f}  ratio_c={ratio_c:.3f}  {status}")

        print(f"  Summary: user passed {n_user_pass}/{n_tests}, "
              f"corrected passed {n_corr_pass}/{n_tests}")


def single_pass():
    """Check that σ_td is large enough for meaningful test."""
    np.random.seed(123)
    duration, imu_freq = 5.0, 400
    imu_ts, R_gt, v_gt, P_gt, gyro, acc = generate_imu(duration, imu_freq)
    ric = exp_so3(np.array([0.02, -0.01, 0.03]))
    tic = np.array([0.05, 0.01, -0.02])
    K = np.array([[400., 0, 320.], [0, 400., 240.], [0, 0, 1.]])

    eps = 1e-5
    td_true = 0.015

    # Pick a pair
    tc_i, tc_j = 1.0, 1.5

    # Get spread landmarks at true poses
    lms = generate_spread_landmarks(
        25, K, tc_i, tc_j, td_true,
        imu_ts, R_gt, v_gt, P_gt, gyro, acc, ric, tic)
    n_pts = len(lms)

    R_i, v_i, P_i, w_i_b, a_i_b = imu_at(
        imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_i - td_true)
    R_j, v_j, P_j, w_j_b, a_j_b = imu_at(
        imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_j - td_true)

    # Print per-landmark details
    print(f"\n{'lm':>4s}  {'pt_C':>30s}  {'depth':>7s}  "
          f"{'|J_fd|':>8s}  {'|J_corr|':>8s}  {'|J_user|':>8s}  "
          f"{'ang_c':>7s}  {'ang_u':>7s}")
    print("-" * 105)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dt_acc = 0.001

    for k, (pt_C, p_C_j_meas) in enumerate(lms):
        p_G = R_i @ (ric @ pt_C + tic) + P_i
        p_I_j = R_j.T @ (p_G - P_j)
        p_C_j = ric.T @ (p_I_j - tic)
        z = p_C_j[2]
        reduce = np.array([[fx / z, 0, -fx * p_C_j[0] / (z * z)],
                           [0, fy / z, -fy * p_C_j[1] / (z * z)]])

        Ju = jac_user(pt_C, R_i, R_j, P_i, P_j, w_i_b, w_j_b, v_i, v_j,
                      a_i_b, a_j_b, ric, tic, reduce, dt_acc)
        Jc = jac_corrected(pt_C, R_i, R_j, P_i, P_j, w_i_b, w_j_b, v_i, v_j,
                           a_i_b, a_j_b, ric, tic, reduce, dt_acc)

        # FD for this single landmark
        def r_lm(td_s):
            R_i_s, _, P_i_s, _, _ = imu_at(
                imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_i - td_true + td_s)
            R_j_s, _, P_j_s, _, _ = imu_at(
                imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc_j - td_true + td_s)
            p_G_s = R_i_s @ (ric @ pt_C + tic) + P_i_s
            p_I_j_s = R_j_s.T @ (p_G_s - P_j_s)
            p_C_j_s = ric.T @ (p_I_j_s - tic)
            z_s = p_C_j_s[2]
            uv_s = np.array([fx * p_C_j_s[0] / z_s + cx,
                             fy * p_C_j_s[1] / z_s + cy])
            z_m = p_C_j_meas[2]
            uv_m = np.array([fx * p_C_j_meas[0] / z_m + cx,
                             fy * p_C_j_meas[1] / z_m + cy])
            return uv_m - uv_s

        Jf = (r_lm(-eps) - r_lm(+eps)) / (2 * eps)  # r(-eps) = r(td+eps), r(+eps) = r(td-eps)

        cos_c = np.dot(Jc, Jf) / (norm(Jc) * norm(Jf) + 1e-15)
        cos_u = np.dot(Ju, Jf) / (norm(Ju) * norm(Jf) + 1e-15)
        ang_c = np.degrees(np.arccos(np.clip(cos_c, -1, 1)))
        ang_u = np.degrees(np.arccos(np.clip(cos_u, -1, 1)))

        # Also compute the min singular value of the 2×1 "information" for
        # this landmark to check degeneracy
        sigma = norm(Jf)

        print(f"  {k:3d}  ({pt_C[0]:8.3f},{pt_C[1]:8.3f},{pt_C[2]:8.3f})  "
              f"{pt_C[2]:6.2f}m  {sigma:8.4f}  {norm(Jc):8.4f}  {norm(Ju):8.4f}  "
              f"{ang_c:5.1f}deg  {ang_u:5.1f}deg")


# ── Old global-coordinate formula verification ──

def verify_old_formula():
    """Verify the old single-frame global-coordinate formula.

    Scenario: single camera frame, known global landmarks.
    Residual = pixel error when projecting p_G through the camera at t_cam - t_d.
    """
    np.random.seed(123)
    duration, imu_freq = 5.0, 400
    imu_ts, R_gt, v_gt, P_gt, gyro, acc = generate_imu(duration, imu_freq)
    ric = exp_so3(np.array([0.02, -0.01, 0.03]))
    tic = np.array([0.05, 0.01, -0.02])
    K = np.array([[400., 0, 320.], [0, 400., 240.], [0, 0, 1.]])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    dt_acc = 0.001
    eps = 1e-5
    n_lm = 25

    print("\n" + "=" * 100)
    print("OLD global-coordinate formula verification (single frame, known p_G)")
    print(f"  Landmarks per test: {n_lm}")
    print("=" * 100)

    for td_true in [0.0, 0.004, 0.015, 0.030]:
        print(f"\n─── t_d_true = {td_true * 1000:.0f} ms ───")
        n_tests = 0
        n_pass = 0

        for _ in range(30):
            tc = np.random.uniform(0.5, duration - 0.5)

            # Query IMU at true exposure time
            R, v, P, w_body, a_body = imu_at(
                imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc - td_true)

            # Generate spread landmarks (as global points)
            n_grid = int(np.ceil(np.sqrt(n_lm)))
            landmarks_global = []
            for gx in range(n_grid):
                for gy in range(n_grid):
                    if len(landmarks_global) >= n_lm:
                        break
                    u = (gx + 0.2 + 0.6 * np.random.random()) / n_grid * 640
                    v = (gy + 0.2 + 0.6 * np.random.random()) / n_grid * 480
                    depth = 2.0 + 8.0 * (gx * n_grid + gy) / (n_grid * n_grid)
                    pt_C = np.array([(u - cx) * depth / fx,
                                     (v - cy) * depth / fy,
                                     depth])
                    Rc = R @ ric
                    Pc = P + R @ tic
                    p_G = Rc @ pt_C + Pc
                    landmarks_global.append(p_G)
                if len(landmarks_global) >= n_lm:
                    break

            if len(landmarks_global) < 8:
                continue
            n_tests += 1
            n_pts = len(landmarks_global)

            # Build stacked Jacobians
            J_old_all = np.zeros((2 * n_pts, 1))

            def residuals_stacked_old(td_shift):
                R_s, _, P_s, _, _ = imu_at(
                    imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc - td_true + td_shift)
                r_all = np.zeros(2 * n_pts)
                for k, p_G in enumerate(landmarks_global):
                    p_I = R_s.T @ (p_G - P_s)
                    p_C = ric.T @ (p_I - tic)
                    z_s = p_C[2]
                    uv_s = np.array([fx * p_C[0] / z_s + cx,
                                     fy * p_C[1] / z_s + cy])
                    # Measurement at true pose (td_shift = 0)
                    R_t, _, P_t, _, _ = imu_at(
                        imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc - td_true)
                    p_I_t = R_t.T @ (p_G - P_t)
                    p_C_t = ric.T @ (p_I_t - tic)
                    z_t = p_C_t[2]
                    uv_m = np.array([fx * p_C_t[0] / z_t + cx,
                                     fy * p_C_t[1] / z_t + cy])
                    r_all[2 * k:2 * k + 2] = uv_m - uv_s
                return r_all

            # Compute stacked analytical Jacobian
            for k, p_G in enumerate(landmarks_global):
                p_I = R.T @ (p_G - P)
                p_C = ric.T @ (p_I - tic)
                z = p_C[2]
                reduce = np.array([[fx / z, 0, -fx * p_C[0] / (z * z)],
                                   [0, fy / z, -fy * p_C[1] / (z * z)]])

                J_old_all[2 * k:2 * k + 2, :] = jac_old_exact(
                    p_G, R, P, w_body, v, a_body, ric, reduce, dt_acc).reshape(2, 1)

            # FD
            r_plus = residuals_stacked_old(+eps)   # query later  → t_d smaller
            r_minus = residuals_stacked_old(-eps)  # query earlier → t_d larger
            J_fd_all = (r_minus - r_plus) / (2 * eps)

            # Cosine similarity
            def cos_sim(a, b):
                return np.dot(a.ravel(), b.ravel()) / (norm(a) * norm(b) + 1e-15)

            cos_o = cos_sim(J_old_all, J_fd_all)
            ang_o = np.degrees(np.arccos(np.clip(cos_o, -1, 1)))
            ratio_o = norm(J_old_all) / (norm(J_fd_all) + 1e-15)

            # Per-landmark angles
            per_ang = []
            for k in range(n_pts):
                jo = J_old_all[2*k:2*k+2].ravel()
                jf = J_fd_all[2*k:2*k+2].ravel()
                c = np.dot(jo, jf) / (norm(jo) * norm(jf) + 1e-15)
                per_ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))

            per_mean = np.mean(per_ang)
            status = f"OK  (per={per_mean:.1f}deg)" if abs(cos_o) > 0.999 else f"FAIL (per={per_mean:.1f}deg)"
            if abs(cos_o) > 0.999:
                n_pass += 1

            print(f"  [{n_tests:2d}] n_lm={n_pts:2d}  "
                  f"stacked_ang={ang_o:6.1f}deg  ratio={ratio_o:.3f}  {status}")

        print(f"  Summary: old formula passed {n_pass}/{n_tests}")


if __name__ == "__main__":
    single_pass()
    print("\n" + "=" * 100)
    main()
    verify_old_formula()
