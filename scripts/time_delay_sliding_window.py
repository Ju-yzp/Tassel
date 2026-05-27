"""
Sliding-window VIO with time delay estimation (absolute IMU pose method).

Design:
- Stereo depth → absolute 3D point in camera frame
- 2D pixel reprojection residual between frame pairs
- Shared t_d per sliding window
- IMU poses integrated from t_cam to t_cam + t_d via small midpoint steps
  (step = 2/3 * IMU sampling interval)
- Direction probing: try +dt first, fall back to -dt if residual increases
- No free pose optimization — poses come purely from IMU integration
"""

import numpy as np
from numpy.linalg import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'lines.linewidth': 1.0, 'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# ═══════════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════════
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(phi):
    th = norm(phi)
    if th < 1e-12:
        return np.eye(3)
    a = phi / th
    return np.eye(3) + np.sin(th) * skew(a) + (1 - np.cos(th)) * skew(a) @ skew(a)


def log_so3(R):
    ct = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(ct)
    if th < 1e-12:
        return np.zeros(3)
    st = np.sin(th)
    if abs(st) < 1e-12:
        A = (R - np.eye(3)) / 2
        v = np.array([A[2, 1], A[0, 2], A[1, 0]])
        s = norm(v)
        if s < 1e-12:
            w, V = np.linalg.eigh(R)
            v = V[:, np.argmax(w)] * th
            return v
        return v / s * th
    return th / (2 * st) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


# ═══════════════════════════════════════════════════════════════════
# IMU trajectory
# ═══════════════════════════════════════════════════════════════════
def imu_at(imu_ts, R_gt, v_gt, P_gt, gyro_body, acc_body, t):
    """Single-step midpoint query at time t."""
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


def integrate_imu_from_to(imu_ts, gyro, acc, R_gt, v_gt, P_gt,
                           t_start, t_target, dt_step):
    """Small-step midpoint integration from t_start to t_target.

    Each step <= dt_step (= 2/3 * imu_dt for stability).
    Returns (R, v, P, w, a) at t_target, or None on numerical failure.
    """
    R, v, P, w_cur, a_cur = imu_at(imu_ts, R_gt, v_gt, P_gt, gyro, acc, t_start)

    if abs(t_target - t_start) < 1e-12:
        return R, v, P, w_cur, a_cur

    direction = 1 if t_target > t_start else -1
    t = t_start

    while direction * (t_target - t) > 1e-12:
        dt_remaining = t_target - t
        dt_actual = direction * min(dt_step, abs(dt_remaining))
        t_next = t + dt_actual

        # Interpolate gyro/acc at t_next from raw IMU data
        idx = np.searchsorted(imu_ts, t_next)
        if idx <= 0:
            w_next, a_next = gyro[0].copy(), acc[0].copy()
        elif idx >= len(imu_ts):
            w_next, a_next = gyro[-1].copy(), acc[-1].copy()
        else:
            i0, i1 = idx - 1, idx
            alpha = (t_next - imu_ts[i0]) / (imu_ts[i1] - imu_ts[i0])
            w_next = (1 - alpha) * gyro[i0] + alpha * gyro[i1]
            a_next = (1 - alpha) * acc[i0] + alpha * acc[i1]

        # Midpoint integration
        w_mid = 0.5 * (w_cur + w_next)
        a_mid = 0.5 * (a_cur + a_next)

        R_new = R @ exp_so3(w_mid * dt_actual)
        v_new = v + R @ a_mid * dt_actual
        P_new = P + v * dt_actual + 0.5 * R @ a_mid * dt_actual ** 2

        if np.any(~np.isfinite(R_new)) or np.any(~np.isfinite(P_new)):
            return None

        R, v, P = R_new, v_new, P_new
        w_cur, a_cur = w_next, a_next
        t = t_next

    return R, v, P, w_cur, a_cur


def generate_imu(duration, freq):
    dt = 1.0 / freq
    n = int(duration / dt)
    ts = np.arange(n) * dt
    gyro = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    for i in range(n):
        t = ts[i]
        gyro[i] = [5.0 * np.sin(2 * np.pi * 0.8 * t),
                   4.0 * np.cos(2 * np.pi * 1.1 * t + 1.0),
                   3.0 * np.sin(2 * np.pi * 1.5 * t + 0.3)]
        acc[i] = [8.0 * np.cos(2 * np.pi * 0.5 * t + 0.7),
                  6.0 * np.sin(2 * np.pi * 0.7 * t),
                  3.0 * np.cos(2.0 * np.pi * t) + 9.81]
    R = np.zeros((n, 3, 3)); R[0] = np.eye(3)
    v = np.zeros((n, 3)); P = np.zeros((n, 3))
    for i in range(n - 1):
        gm = 0.5 * (gyro[i] + gyro[i + 1])
        am = 0.5 * (acc[i] + acc[i + 1])
        R[i + 1] = R[i] @ exp_so3(gm * dt)
        v[i + 1] = v[i] + R[i] @ am * dt
        P[i + 1] = P[i] + v[i] * dt + 0.5 * R[i] @ am * dt ** 2
    return ts, R, v, P, gyro, acc


# ═══════════════════════════════════════════════════════════════════
# Landmarks & observations
# ═══════════════════════════════════════════════════════════════════
def generate_landmarks(imu_ts, R_gt, P_gt, ric, tic, K, n_lm, duration):
    sample_ts = np.linspace(0.5, duration - 0.5, 10)
    lms = []
    while len(lms) < n_lm:
        tc = np.random.choice(sample_ts)
        idx = np.searchsorted(imu_ts, tc)
        Rc = R_gt[idx] @ ric
        Pc = P_gt[idx] + R_gt[idx] @ tic
        depth = np.random.uniform(1.5, 15)
        u = np.random.uniform(60, 580)
        v = np.random.uniform(40, 440)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p_C = np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth])
        p_G = Rc @ p_C + Pc
        vis = 0
        for ts in sample_ts:
            R, _, P, _, _ = imu_at(imu_ts, R_gt, np.zeros(len(imu_ts)),
                                     P_gt, np.zeros((len(imu_ts), 3)),
                                     np.zeros((len(imu_ts), 3)), ts)
            Rc2, Pc2 = R @ ric, P + R @ tic
            pc = Rc2.T @ (p_G - Pc2)
            if pc[2] <= 0.1: continue
            uv = K[:2, :2] @ (pc[:2] / pc[2]) + K[:2, 2]
            if 15 < uv[0] < 625 and 15 < uv[1] < 465: vis += 1
        if vis >= 3: lms.append(p_G)
    return lms


def gen_obs(cam_ts, imu_ts, R_gt, v_gt, P_gt, gyro, acc, ric, tic, K,
            landmarks, td_arr, dt_step):
    """Per-frame observations at true exposure times t_cam + td_arr[k]."""
    obs = []
    for k, tc in enumerate(cam_ts):
        result = integrate_imu_from_to(imu_ts, gyro, acc, R_gt, v_gt, P_gt,
                                        tc, tc + td_arr[k], dt_step)
        if result is None:
            R, _, P, _, _ = imu_at(imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc)
        else:
            R, _, P, _, _ = result
        Rc, Pc = R @ ric, P + R @ tic
        frame = []
        for lm_id, p_G in enumerate(landmarks):
            p_C = Rc.T @ (p_G - Pc)
            if p_C[2] <= 0.1: continue
            uv = K[:2, :2] @ (p_C[:2] / p_C[2]) + K[:2, 2]
            if not (15 < uv[0] < 625 and 15 < uv[1] < 465): continue
            frame.append((lm_id, p_C.copy(), uv.copy()))
        obs.append(frame)
    return obs


def build_edges(obs_win):
    lm2f = defaultdict(dict)
    for fi, frame in enumerate(obs_win):
        for lm_id, p_C, uv in frame:
            lm2f[lm_id][fi] = (p_C, uv)
    edges = []
    for lm_id, frames in lm2f.items():
        fl = sorted(frames.keys())
        for a in range(len(fl)):
            for b in range(a + 1, len(fl)):
                i, j = fl[a], fl[b]
                p_C_i, _ = frames[i]
                _, uv_j = frames[j]
                edges.append([i, j, *p_C_i, *uv_j])
    if not edges:
        return np.zeros((0, 8))
    return np.array(edges)


# ═══════════════════════════════════════════════════════════════════
# Solver: IMU-only poses, probe direction for t_d
# ═══════════════════════════════════════════════════════════════════
class SlidingWindowSolver:
    def __init__(self, imu_ts, gyro, acc, R_gt, v_gt, P_gt, ric, tic, K, dt_step):
        self.imu_ts = imu_ts; self.gyro = gyro; self.acc = acc
        self.R_gt = R_gt; self.v_gt = v_gt; self.P_gt = P_gt
        self.ric = ric; self.tic = tic
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.dt_step = dt_step

    def _imu_poses(self, ts_win, td):
        """IMU-integrated pose array (N x 6) at t_cam + td, or None on failure."""
        N = len(ts_win)
        poses = np.zeros((N, 6))
        for k in range(N):
            result = integrate_imu_from_to(
                self.imu_ts, self.gyro, self.acc,
                self.R_gt, self.v_gt, self.P_gt,
                ts_win[k], ts_win[k] + td, self.dt_step)
            if result is None:
                return None
            R, v, P, w, a = result
            poses[k] = np.concatenate([log_so3(R), P])
        return poses

    def _full_state(self, ts_win, td):
        """Integrate all frames and return (poses, states).  States carry the IMU
        (R, v, P, w, a) at each frame's t_cam + td so increments can be applied
        by right-multiplication without re-integrating from t_cam."""
        N = len(ts_win)
        poses = np.zeros((N, 6))
        states = {}
        for k in range(N):
            result = integrate_imu_from_to(
                self.imu_ts, self.gyro, self.acc,
                self.R_gt, self.v_gt, self.P_gt,
                ts_win[k], ts_win[k] + td, self.dt_step)
            if result is None:
                return None, None
            R, v, P, w, a = result
            poses[k] = np.concatenate([log_so3(R), P])
            states[k] = (R, v, P, w, a)
        return poses, states

    def _increment_poses(self, ts_win, states, dt):
        """Right-multiplication pose update: apply a single midpoint step of size
        dt to each frame's stored IMU state.  No re-integration from camera time.

        Returns (new_poses, new_states) or (None, None) on failure.
        """
        N = len(ts_win)
        new_poses = np.zeros((N, 6))
        new_states = {}
        for k in range(N):
            R, v, P, w_cur, a_cur = states[k]
            t_cur = ts_win[k] + self._states_td
            t_target = t_cur + dt

            # Interpolate gyro/acc at t_target from raw IMU data (single query)
            idx = np.searchsorted(self.imu_ts, t_target)
            if idx <= 0:
                w_next, a_next = self.gyro[0].copy(), self.acc[0].copy()
            elif idx >= len(self.imu_ts):
                w_next, a_next = self.gyro[-1].copy(), self.acc[-1].copy()
            else:
                i0, i1 = idx - 1, idx
                alpha = (t_target - self.imu_ts[i0]) / (self.imu_ts[i1] - self.imu_ts[i0])
                w_next = (1 - alpha) * self.gyro[i0] + alpha * self.gyro[i1]
                a_next = (1 - alpha) * self.acc[i0] + alpha * self.acc[i1]

            # Single midpoint step — right-multiplication on SE(3)
            w_mid = 0.5 * (w_cur + w_next)
            a_mid = 0.5 * (a_cur + a_next)

            R_new = R @ exp_so3(w_mid * dt)
            v_new = v + R @ a_mid * dt
            P_new = P + v * dt + 0.5 * R @ a_mid * dt ** 2

            if np.any(~np.isfinite(R_new)) or np.any(~np.isfinite(P_new)):
                return None, None

            new_poses[k] = np.concatenate([log_so3(R_new), P_new])
            new_states[k] = (R_new, v_new, P_new, w_next, a_next)

        return new_poses, new_states

    def _compute_cost(self, poses_arr, edges):
        """Total squared reprojection error."""
        if poses_arr is None or len(edges) == 0:
            return float('inf')
        E_i = edges[:, 0].astype(int)
        E_j = edges[:, 1].astype(int)
        E_pCi = edges[:, 2:5]
        E_uvj = edges[:, 5:7]
        total = 0.0
        for k in range(len(edges)):
            i, j = E_i[k], E_j[k]
            p_C_i = E_pCi[k]
            phi_i = poses_arr[i, :3]; P_i_est = poses_arr[i, 3:6]
            phi_j = poses_arr[j, :3]; P_j_est = poses_arr[j, 3:6]
            R_i_est = exp_so3(phi_i); R_j_est = exp_so3(phi_j)
            p_G = R_i_est @ (self.ric @ p_C_i + self.tic) + P_i_est
            p_I_j = R_j_est.T @ (p_G - P_j_est)
            p_C_j_pred = self.ric.T @ (p_I_j - self.tic)
            z = p_C_j_pred[2]
            uv_pred = np.array([self.fx * p_C_j_pred[0] / z + self.cx,
                                self.fy * p_C_j_pred[1] / z + self.cy])
            r = E_uvj[k] - uv_pred
            total += np.dot(r, r)
        return total

    def solve(self, ts_win, obs_win, x0_td, fix_td):
        edges = build_edges(obs_win)
        n_edges = len(edges)
        if n_edges == 0:
            imu_poses, _ = self._full_state(ts_win, x0_td)
            if imu_poses is None:
                imu_poses = np.zeros((len(ts_win), 6))
            return imu_poses, x0_td, 0.0, [x0_td]

        td = x0_td
        td_history = [td]
        step_schedule = [0.0025, 0.001, 0.0005, 0.0001]

        # Full integration once to get initial poses and IMU states
        imu_poses, states = self._full_state(ts_win, td)
        if imu_poses is None:
            return np.zeros((len(ts_win), 6)), td, 0.0, td_history
        self._states_td = td

        for probe_step in step_schedule:
            if fix_td:
                break
            for it in range(100):
                cost_cur = self._compute_cost(imu_poses, edges)

                accepted = False

                # Try positive direction — single right-multiplication step
                td_pos = np.clip(td + probe_step, -0.1, 0.1)
                dt_pos = td_pos - self._states_td
                inc_poses, inc_states = self._increment_poses(ts_win, states, dt_pos)
                if inc_poses is not None:
                    cost_pos = self._compute_cost(inc_poses, edges)
                    if cost_pos < cost_cur:
                        td = td_pos
                        imu_poses, states = inc_poses, inc_states
                        self._states_td = td_pos
                        td_history.append(td)
                        accepted = True

                if not accepted:
                    td_neg = np.clip(td - probe_step, -0.1, 0.1)
                    dt_neg = td_neg - self._states_td
                    inc_poses, inc_states = self._increment_poses(ts_win, states, dt_neg)
                    if inc_poses is not None:
                        cost_neg = self._compute_cost(inc_poses, edges)
                        if cost_neg < cost_cur:
                            td = td_neg
                            imu_poses, states = inc_poses, inc_states
                            self._states_td = td_neg
                            td_history.append(td)
                            accepted = True

                if not accepted:
                    break

        return imu_poses, td, np.sqrt(
            self._compute_cost(imu_poses, edges) / max(n_edges, 1)), td_history


def run_sliding_window(cam_ts, observations, window_size,
                       imu_ts, gyro, acc, R_gt, v_gt, P_gt,
                       ric, tic, K, dt_step, fix_td):
    solver = SlidingWindowSolver(imu_ts, gyro, acc, R_gt, v_gt, P_gt,
                                  ric, tic, K, dt_step)
    nf = len(cam_ts)

    all_poses = np.zeros((nf, 6))
    all_tds = np.zeros(nf)
    all_rms = np.zeros(nf)
    td_current = 0.0
    td_iter_history = None

    for start in range(0, nf - window_size + 1):
        end = start + window_size
        p_opt, td_opt, rms, td_hist = solver.solve(
            cam_ts[start:end], observations[start:end], td_current, fix_td)
        all_poses[start:end] = p_opt
        all_tds[start:end] = td_opt
        all_rms[start:end] = rms
        td_current = td_opt
        if start == 0:
            td_iter_history = td_hist
            edges = build_edges(observations[start:end])
            print(f"    [win 0] td={td_opt*1000:.1f}ms, rms={rms:.4f}px, "
                  f"edges={len(edges)}, iters={len(td_hist)-1}")

    return all_poses, all_tds, all_rms, td_iter_history


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    duration, imu_freq, cam_freq, win_size = 4.0, 400, 10, 4
    dt_step = (2.0 / 3.0) / imu_freq  # 2/3 of IMU sampling interval
    ric = exp_so3(np.array([0.02, -0.01, 0.03]))
    tic = np.array([0.05, 0.01, -0.02])
    K = np.array([[400., 0, 320.], [0, 400., 240.], [0, 0, 1.]])

    print(f"IMU: {imu_freq} Hz, dt_step: {dt_step*1000:.3f} ms (2/3 of imu_dt)")
    print("Generating IMU trajectory...")
    imu_ts, R_gt, v_gt, P_gt, gyro, acc = generate_imu(duration, imu_freq)
    cam_ts_all = np.arange(0.5, duration - 0.1, 1.0 / cam_freq)
    print(f"  IMU: {len(imu_ts)} samples, Camera: {len(cam_ts_all)} frames")

    print("Generating landmarks...")
    landmarks = generate_landmarks(imu_ts, R_gt, P_gt, ric, tic, K, 15, duration)
    obs_check = gen_obs(cam_ts_all, imu_ts, R_gt, v_gt, P_gt, gyro, acc,
                         ric, tic, K, landmarks, np.zeros(len(cam_ts_all)), dt_step)
    valid = [k for k in range(len(cam_ts_all)) if len(obs_check[k]) >= 5]
    cam_ts = cam_ts_all[valid[0]:valid[-1] + 1]
    print(f"  Landmarks: {len(landmarks)}, Valid frames: {len(cam_ts)}")

    true_delays = [-0.0111, 0.004, 0.015, 0.02356, 0.030]
    results = {}
    np.random.seed(42)

    for td_true in true_delays:
        # Scale jitter with base t_d: 10% of |td|, clamped to [0.8, 3.0] ms
        jitter_amp = np.clip(abs(td_true) * 0.1, 0.0008, 0.003)
        td_jitter = np.random.uniform(-jitter_amp, jitter_amp, len(cam_ts))
        print(f"\n─── base t_d = {td_true * 1000:.2f} ms, jitter = ±{jitter_amp*1000:.1f} ms ───")
        td_per_frame = np.full(len(cam_ts), td_true) + td_jitter
        obs = gen_obs(cam_ts, imu_ts, R_gt, v_gt, P_gt, gyro, acc,
                      ric, tic, K, landmarks, td_per_frame, dt_step)

        print("  Without t_d correction...")
        p_no, _, rms_no, _ = run_sliding_window(cam_ts, obs, win_size,
                                      imu_ts, gyro, acc, R_gt, v_gt, P_gt,
                                      ric, tic, K, dt_step, fix_td=True)
        print("  With t_d estimation...")
        p_yes, t_est, rms_yes, td_hist = run_sliding_window(cam_ts, obs, win_size,
                                           imu_ts, gyro, acc, R_gt, v_gt, P_gt,
                                           ric, tic, K, dt_step, fix_td=False)

        # Ground truth IMU poses at per-frame true exposure times
        gt = np.zeros((len(cam_ts), 6))
        for k, tc in enumerate(cam_ts):
            result = integrate_imu_from_to(imu_ts, gyro, acc, R_gt, v_gt, P_gt,
                                            tc, tc + td_per_frame[k], dt_step)
            if result is None:
                R, _, P, _, _ = imu_at(imu_ts, R_gt, v_gt, P_gt, gyro, acc, tc)
            else:
                R, _, P, _, _ = result
            gt[k] = np.concatenate([log_so3(R), P])

        err_no = norm(p_no[:, 3:] - gt[:, 3:], axis=1) * 100
        err_yes = norm(p_yes[:, 3:] - gt[:, 3:], axis=1) * 100

        print(f"  t_d est: last5={np.mean(t_est[-5:])*1000:.2f} ms, "
              f"mean={t_est.mean()*1000:.2f} ms, final={t_est[-1]*1000:.2f} ms")
        print(f"  Pos err (mean): {err_no.mean():.2f} → {err_yes.mean():.2f} cm")
        print(f"  Pos err (final): {err_no[-1]:.2f} → {err_yes[-1]:.2f} cm")
        print(f"  RMS px (mean): {rms_no.mean():.4f} → {rms_yes.mean():.4f} px")

        results[td_true] = {'tds': t_est, 'err_no': err_no, 'err_yes': err_yes,
                           'rms_no': rms_no, 'rms_yes': rms_yes,
                           'gt_poses': gt, 'p_no': p_no, 'p_yes': p_yes,
                           'td_hist': td_hist}

    # ── Plots ──
    colors = ['#2166ac', '#4daf4a', '#b2182b', '#ff7f00', '#984ea3']
    n_cases = len(true_delays)

    fig1, ax1 = plt.subplots(figsize=(8, 3.5))
    for idx, td in enumerate(true_delays):
        h = results[td]['td_hist']
        ax1.plot(range(len(h)), np.array(h) * 1000, color=colors[idx], lw=1.0,
                 marker='.', markersize=3,
                 label=f'true = {td * 1000:.2f} ms')
        ax1.axhline(td * 1000, color=colors[idx], ls='--', lw=0.7, alpha=0.6)
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('$t_d$ (ms)')
    ax1.set_title('$t_d$ convergence per iteration (first window)')
    ax1.legend(fontsize=7)
    fig1.tight_layout()
    fig1.savefig("/home/adrewn/Tassel/scripts/td_sw_convergence.png")

    n_cols = 3
    n_rows = 2
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(11, 6.5), sharey=True)
    for idx, td in enumerate(true_delays):
        r = results[td]
        ax = axes[idx // n_cols, idx % n_cols]
        ax.plot(r['err_no'], color='#d7191c', lw=0.8, label='No $t_d$')
        ax.plot(r['err_yes'], color='#2b83ba', lw=0.8, label='With $t_d$')
        ax.set_title(f'$t_d^{{true}} = {td*1000:.2f}$ ms')
        ax.set_xlabel('Frame')
        if idx == 0: ax.set_ylabel('Pos error (cm)')
        ax.legend(fontsize=6)
    for idx in range(n_cases, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig2.tight_layout()
    fig2.savefig("/home/adrewn/Tassel/scripts/td_sw_position_error.png")

    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(11, 6.5), sharey=True)
    for idx, td in enumerate(true_delays):
        r = results[td]
        ax = axes[idx // n_cols, idx % n_cols]
        ax.plot(r['rms_no'], color='#d7191c', lw=0.8, label='No $t_d$')
        ax.plot(r['rms_yes'], color='#2b83ba', lw=0.8, label='With $t_d$')
        ax.set_title(f'$t_d^{{true}} = {td*1000:.2f}$ ms')
        ax.set_xlabel('Frame')
        if idx == 0: ax.set_ylabel('RMS px')
        ax.legend(fontsize=6)
    for idx in range(n_cases, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig3.tight_layout()
    fig3.savefig("/home/adrewn/Tassel/scripts/td_sw_reproj_error.png")

    # ── Figure 4: 3D trajectory + sampled camera poses ──
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    n_3d_cols = 3
    n_3d_rows = 2
    fig4 = plt.figure(figsize=(13, 8))
    for idx, td in enumerate(true_delays):
        r = results[td]
        gt_poses = r['gt_poses']
        cam_wrong = np.zeros((len(cam_ts), 3))
        cam_correct = np.zeros((len(cam_ts), 3))
        cam_est = np.zeros((len(cam_ts), 3))
        for k in range(len(cam_ts)):
            cam_wrong[k] = r['p_no'][k, 3:] + exp_so3(r['p_no'][k, :3]) @ tic
            cam_correct[k] = gt_poses[k, 3:] + exp_so3(gt_poses[k, :3]) @ tic
            cam_est[k] = r['p_yes'][k, 3:] + exp_so3(r['p_yes'][k, :3]) @ tic

        step = max(1, len(cam_ts) // 18)
        ax = fig4.add_subplot(n_3d_rows, n_3d_cols, idx + 1, projection='3d')
        ax.plot(P_gt[::40, 0], P_gt[::40, 1], P_gt[::40, 2],
                color='gray', alpha=0.2, lw=0.4)
        ax.scatter(cam_wrong[::step, 0], cam_wrong[::step, 1], cam_wrong[::step, 2],
                   c='#d7191c', s=20, marker='x', label='Wrong ($t_d=0$)', zorder=5)
        ax.scatter(cam_correct[::step, 0], cam_correct[::step, 1], cam_correct[::step, 2],
                   c='#2b83ba', s=20, marker='o', label='Correct (true $t_d$)', zorder=5)
        ax.scatter(cam_est[::step, 0], cam_est[::step, 1], cam_est[::step, 2],
                   c='#4daf4a', s=14, marker='^', label='Estimated $t_d$', zorder=5)
        for i in range(0, len(cam_ts), step):
            ax.plot([cam_wrong[i, 0], cam_correct[i, 0]],
                    [cam_wrong[i, 1], cam_correct[i, 1]],
                    [cam_wrong[i, 2], cam_correct[i, 2]],
                    color='#fdae61', lw=0.5, alpha=0.5)
        ax.scatter(*cam_wrong[0], c='#d7191c', s=40, marker='x', zorder=6)
        ax.set_title(f'$t_d^{{true}} = {td*1000:.2f}$ ms', fontsize=8)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.legend(fontsize=5.5, loc='upper left')
        pts = np.vstack([cam_wrong[::step], cam_correct[::step], P_gt[::40]])
        ranges = pts.max(axis=0) - pts.min(axis=0)
        mid = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
        rng = 0.6 * ranges.max()
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)
    # hide extra subplot
    for idx in range(n_cases, n_3d_rows * n_3d_cols):
        axes_flat = fig4.axes
        for ax in axes_flat:
            pass
        fig4.delaxes(fig4.axes[-1])
    fig4.tight_layout()
    fig4.savefig("/home/adrewn/Tassel/scripts/td_sw_3d_poses.png")

    # ── Figure 5: Camera pose triad comparison (largest |t_d| case, 3 keyframes) ──
    td_extreme = max(true_delays, key=abs)
    r_ext = results[td_extreme]
    gt_ext = r_ext['gt_poses']
    cam_w_ext = np.zeros((len(cam_ts), 3, 3))
    cam_c_ext = np.zeros((len(cam_ts), 3, 3))
    cam_pw_ext = np.zeros((len(cam_ts), 3))
    cam_pc_ext = np.zeros((len(cam_ts), 3))
    for k in range(len(cam_ts)):
        R_w = exp_so3(r_ext['p_no'][k, :3])
        R_c = exp_so3(gt_ext[k, :3])
        cam_w_ext[k] = R_w @ ric
        cam_c_ext[k] = R_c @ ric
        cam_pw_ext[k] = r_ext['p_no'][k, 3:] + R_w @ tic
        cam_pc_ext[k] = gt_ext[k, 3:] + R_c @ tic

    key_frames = [len(cam_ts)//4, len(cam_ts)//2, 3*len(cam_ts)//4]
    axis_len = 0.18
    axis_colors = ['#e41a1c', '#4daf4a', '#377eb8']

    def draw_cam_axes(ax, pos, rot, alpha_val, scale=1.0):
        al = axis_len * scale
        for d, c in enumerate(axis_colors):
            end = pos + rot[:, d] * al
            ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                    color=c, lw=1.5 * scale, alpha=alpha_val)

    fig5 = plt.figure(figsize=(14, 5))
    for sub_idx, kf in enumerate(key_frames):
        ax = fig5.add_subplot(1, 3, sub_idx + 1, projection='3d')
        t_cam_kf = cam_ts[kf]
        t_win = 0.3
        mask = (imu_ts >= t_cam_kf - t_win) & (imu_ts <= t_cam_kf + t_win)
        P_local = P_gt[mask]
        ax.plot(P_local[::5, 0], P_local[::5, 1], P_local[::5, 2],
                color='gray', alpha=0.2, lw=0.5)
        draw_cam_axes(ax, cam_pw_ext[kf], cam_w_ext[kf], 0.55, scale=1.0)
        ax.scatter(*cam_pw_ext[kf], c='#d7191c', s=40, marker='x', zorder=5)
        draw_cam_axes(ax, cam_pc_ext[kf], cam_c_ext[kf], 0.9, scale=1.0)
        ax.scatter(*cam_pc_ext[kf], c='#2b83ba', s=40, marker='o', zorder=5)
        ax.plot([cam_pw_ext[kf][0], cam_pc_ext[kf][0]],
                [cam_pw_ext[kf][1], cam_pc_ext[kf][1]],
                [cam_pw_ext[kf][2], cam_pc_ext[kf][2]],
                color='#fdae61', lw=1.0, alpha=0.7, linestyle='--')
        dist = norm(cam_pw_ext[kf] - cam_pc_ext[kf]) * 100
        ax.set_title(f'Frame {kf} ($t_{{cam}}$={t_cam_kf:.2f}s, '
                     f'$\\|\\Delta P\\|$={dist:.1f}cm)')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        center = 0.5 * (cam_pw_ext[kf] + cam_pc_ext[kf])
        rng = 0.5
        ax.set_xlim(center[0] - rng, center[0] + rng)
        ax.set_ylim(center[1] - rng, center[1] + rng)
        ax.set_zlim(center[2] - rng, center[2] + rng)
    fig5.text(0.12, 0.02, 'Red=X  Green=Y  Blue=Z  |  '
              '— Wrong ($t_d$=0)  — Correct (true $t_d$)  -- Displacement',
              fontsize=7, ha='left', transform=fig5.transFigure)
    fig5.tight_layout(rect=[0, 0.04, 1, 1])
    fig5.savefig("/home/adrewn/Tassel/scripts/td_sw_camera_axes.png")

    print("\nSaved: td_sw_convergence.png, td_sw_position_error.png, "
          "td_sw_reproj_error.png, td_sw_3d_poses.png, td_sw_camera_axes.png")


if __name__ == "__main__":
    main()
