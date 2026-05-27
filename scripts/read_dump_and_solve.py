"""
Read C++ VIO dump files and run sliding-window time delay estimation.

Matches the C++ IMU integration: per-frame state from dump, dual-rotation
midpoint, bias subtraction. Uses same multi-resolution direction probing.

Usage: python3 scripts/read_dump_and_solve.py [dump_prefix] [config_yaml]
"""

import numpy as np
from numpy.linalg import norm
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'lines.linewidth': 1.0, 'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Noto Sans CJK SC', 'DejaVu Sans'],
})

G = np.array([0.0, 0.0, 9.81])


# ═══════════════════════════════════════════════════════════════════
# Geometry helpers
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


def interp_imu(imu_ts, gyro, acc, t):
    """Interpolate raw IMU gyro/acc at time t."""
    idx = np.searchsorted(imu_ts, t)
    if idx <= 0:
        return gyro[0].copy(), acc[0].copy()
    if idx >= len(imu_ts):
        return gyro[-1].copy(), acc[-1].copy()
    i0, i1 = idx - 1, idx
    dt = imu_ts[i1] - imu_ts[i0]
    if dt < 1e-12:
        return gyro[i0].copy(), acc[i0].copy()
    alpha = (t - imu_ts[i0]) / dt
    w = (1 - alpha) * gyro[i0] + alpha * gyro[i1]
    a = (1 - alpha) * acc[i0] + alpha * acc[i1]
    return w, a


# ═══════════════════════════════════════════════════════════════════
# Data readers
# ═══════════════════════════════════════════════════════════════════
def read_imu(path):
    data = np.loadtxt(path, skiprows=1)
    return data[:, 0], data[:, 1:4], data[:, 4:7]


def read_state(path):
    data = np.loadtxt(path, skiprows=1)
    cam_ts = data[:, 0]
    Rs = data[:, 1:10].reshape(-1, 3, 3)
    Ps = data[:, 10:13]
    Vs = data[:, 13:16]
    Bas = data[:, 16:19]
    Bgs = data[:, 19:22]
    return cam_ts, Rs, Ps, Vs, Bas, Bgs


def read_features(path):
    frames = defaultdict(list)
    current_feat = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 5:
                current_feat = {
                    'feat_id': int(parts[0]),
                    'start_frame': int(parts[1]),
                    'depth': float(parts[2]),
                    'tri_source': int(parts[3]),
                    'num_obs': int(parts[4]),
                }
            elif len(parts) == 6 and current_feat is not None:
                frame_id = int(parts[0])
                frames[frame_id].append({
                    'feat_id': current_feat['feat_id'],
                    'u': float(parts[1]), 'v': float(parts[2]),
                    'u_r': float(parts[3]), 'v_r': float(parts[4]),
                    'is_stereo': int(parts[5]) == 1,
                })
    return dict(frames)


# ═══════════════════════════════════════════════════════════════════
# Solver — matches C++ dual-rotation midpoint + bias subtraction
# ═══════════════════════════════════════════════════════════════════
class RealDataSolver:
    def __init__(self, imu_ts, gyro, acc, cam_ts, Rs, Ps, Vs, Bas, Bgs,
                 ric, tic, K, dt_step):
        self.imu_ts = imu_ts
        self.gyro = gyro
        self.acc = acc
        self.cam_ts = cam_ts
        self.Rs = Rs
        self.Ps = Ps
        self.Vs = Vs
        self.Bas = Bas
        self.Bgs = Bgs
        self.ric = ric
        self.tic = tic
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.dt_step = dt_step
        self._num_frames = len(cam_ts)

    def _integrate_step(self, R, v, P, w_cur, a_cur, Ba, Bg, dt, w_next, a_next):
        """One step: C++ dual-rotation midpoint with bias subtraction.
        Matches vio_estimator.cpp lines 78-86 exactly:
          acc_0 = R * (last_imu_acc_ - Ba) - G   (old R, last acc)
          gyr   = 0.5 * (last_imu_gyro_ + imu.gyro) - Bg
          R_new = R * exp(gyr * dt)
          acc_1 = R_new * (imu.acc - Ba) - G     (new R, current acc)
          acc   = 0.5 * (acc_0 + acc_1)
        """
        w_mid = 0.5 * (w_cur + w_next) - Bg
        a_0_w = R @ (a_cur - Ba) - G
        R_new = R @ exp_so3(w_mid * dt)
        a_1_w = R_new @ (a_next - Ba) - G
        a_w = 0.5 * (a_0_w + a_1_w)
        P_new = P + v * dt + 0.5 * a_w * dt * dt
        v_new = v + a_w * dt
        return R_new, v_new, P_new

    def _integrate_from_to(self, t_start, t_target, R0, v0, P0, Ba, Bg):
        """Small-step integration from t_start to t_target, starting from
        state (R0, v0, P0) at t_start. Returns (R, v, P, w_last, a_last)."""
        if abs(t_target - t_start) < 1e-12:
            w0, a0 = interp_imu(self.imu_ts, self.gyro, self.acc, t_start)
            return R0.copy(), v0.copy(), P0.copy(), w0, a0

        direction = 1 if t_target > t_start else -1
        t = t_start
        R, v, P = R0.copy(), v0.copy(), P0.copy()
        w_cur, a_cur = interp_imu(self.imu_ts, self.gyro, self.acc, t)

        while direction * (t_target - t) > 1e-12:
            dt_remaining = t_target - t
            dt_actual = direction * min(self.dt_step, abs(dt_remaining))
            t_next = t + dt_actual
            w_next, a_next = interp_imu(self.imu_ts, self.gyro, self.acc, t_next)

            R, v, P = self._integrate_step(
                R, v, P, w_cur, a_cur, Ba, Bg, dt_actual, w_next, a_next)

            if np.any(~np.isfinite(R)) or np.any(~np.isfinite(P)):
                return None

            w_cur, a_cur = w_next, a_next
            t = t_next

        return R, v, P, w_cur, a_cur

    def _full_state(self, td):
        """Integrate each frame from t_cam to t_cam + td.
        Returns (poses_array, states_dict)."""
        N = self._num_frames
        poses = np.zeros((N, 6))
        states = {}
        for k in range(N):
            result = self._integrate_from_to(
                self.cam_ts[k], self.cam_ts[k] + td,
                self.Rs[k], self.Vs[k], self.Ps[k],
                self.Bas[k], self.Bgs[k])
            if result is None:
                return None, None
            R, v, P, w, a = result
            poses[k] = np.concatenate([log_so3(R), P])
            states[k] = (R, v, P, w, a)
        return poses, states

    def _increment_poses(self, states, delta_td):
        """Right-multiply each frame's state by integrating delta_td further.
        Current states are at t_cam + td_current. delta_td = td_new - td_current."""
        N = self._num_frames
        new_poses = np.zeros((N, 6))
        new_states = {}
        for k in range(N):
            R, v, P, w_cur, a_cur = states[k]
            Ba = self.Bas[k]
            Bg = self.Bgs[k]
            t_cur = self.cam_ts[k] + self._current_td
            t_target = t_cur + delta_td

            direction = 1 if delta_td > 0 else -1
            dt_actual = direction * min(self.dt_step, abs(delta_td))
            t_next = t_cur + dt_actual
            w_next, a_next = interp_imu(self.imu_ts, self.gyro, self.acc, t_next)

            R, v, P = self._integrate_step(
                R, v, P, w_cur, a_cur, Ba, Bg, dt_actual, w_next, a_next)

            if delta_td > dt_actual or -delta_td > dt_actual:
                # Do remaining steps from t_next to t_target
                result = self._integrate_from_to(
                    t_next, t_target, R, v, P, Ba, Bg)
                if result is None:
                    return None, None
                R, v, P, w_next, a_next = result

            if np.any(~np.isfinite(R)) or np.any(~np.isfinite(P)):
                return None, None

            new_poses[k] = np.concatenate([log_so3(R), P])
            new_states[k] = (R, v, P, w_next, a_next)

        return new_poses, new_states

    def _compute_cost(self, poses_arr, edges):
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
            if z < 1e-6:
                continue
            uv_pred = np.array([self.fx * p_C_j_pred[0] / z + self.cx,
                                self.fy * p_C_j_pred[1] / z + self.cy])
            r = E_uvj[k] - uv_pred
            total += np.dot(r, r)
        return total

    def solve(self, obs_win, x0_td, fix_td):
        edges = build_edges(obs_win)
        n_edges = len(edges)
        if n_edges == 0:
            print("    WARNING: no edges")
            imu_poses, _ = self._full_state(x0_td)
            if imu_poses is None:
                imu_poses = np.zeros((self._num_frames, 6))
            return imu_poses, x0_td, 0.0, [x0_td]

        td = x0_td
        td_history = [td]
        step_schedule = [0.0025, 0.001, 0.0005, 0.0001]

        imu_poses, states = self._full_state(td)
        if imu_poses is None:
            return np.zeros((self._num_frames, 6)), td, 0.0, td_history
        self._current_td = td

        for probe_step in step_schedule:
            if fix_td:
                break
            for it in range(100):
                cost_cur = self._compute_cost(imu_poses, edges)
                accepted = False

                td_pos = np.clip(td + probe_step, -0.1, 0.1)
                delta = td_pos - self._current_td
                inc_poses, inc_states = self._increment_poses(states, delta)
                if inc_poses is not None:
                    cost_pos = self._compute_cost(inc_poses, edges)
                    if cost_pos < cost_cur:
                        td = td_pos
                        imu_poses, states = inc_poses, inc_states
                        self._current_td = td_pos
                        td_history.append(td)
                        accepted = True

                if not accepted:
                    td_neg = np.clip(td - probe_step, -0.1, 0.1)
                    delta = td_neg - self._current_td
                    inc_poses, inc_states = self._increment_poses(states, delta)
                    if inc_poses is not None:
                        cost_neg = self._compute_cost(inc_poses, edges)
                        if cost_neg < cost_cur:
                            td = td_neg
                            imu_poses, states = inc_poses, inc_states
                            self._current_td = td_neg
                            td_history.append(td)
                            accepted = True

                if not accepted:
                    break

        return imu_poses, td, np.sqrt(
            self._compute_cost(imu_poses, edges) / max(n_edges, 1)), td_history


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
# Build observations from dump features
# ═══════════════════════════════════════════════════════════════════
def build_observations(features_by_frame, K, baseline):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    all_frame_ids = sorted(features_by_frame.keys())
    if not all_frame_ids:
        return [], []

    frame_id_to_idx = {fid: i for i, fid in enumerate(all_frame_ids)}
    n_frames = len(all_frame_ids)
    obs_win = [[] for _ in range(n_frames)]

    for frame_id, feats in features_by_frame.items():
        win_idx = frame_id_to_idx[frame_id]
        for f in feats:
            u_n, v_n = f['u'], f['v']
            if f['is_stereo'] and f['u_r'] != 0:
                disp = u_n - f['u_r']
                if abs(disp) > 1e-9:
                    depth = baseline / disp
                else:
                    continue
            else:
                continue

            p_C = np.array([u_n * depth, v_n * depth, depth])
            uv_pix = np.array([fx * u_n + cx, fy * v_n + cy])
            obs_win[win_idx].append((f['feat_id'], p_C, uv_pix))

    return obs_win


def parse_yaml_k(path):
    K = np.eye(3)
    in_cam0 = False
    in_intrinsics = False
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('cam0:'):
                in_cam0 = True
            elif in_cam0 and s.startswith('cam1:'):
                break
            elif in_cam0 and s.startswith('intrinsics:'):
                in_intrinsics = True
            elif in_intrinsics and s.startswith('- ['):
                row = [float(x.strip()) for x in s[3:-2].split(',')]
                rows.append(row)
                if len(rows) == 3:
                    K = np.array(rows)
                    break
    return K


def parse_yaml_extrinsics(path):
    import re
    def read_tf(lines, start_idx):
        rows = []
        for i in range(start_idx, min(start_idx + 4, len(lines))):
            m = re.findall(r'-?\d+\.?\d*e?-?\d*', lines[i])
            if len(m) >= 4:
                rows.append([float(x) for x in m[:4]])
            if len(rows) == 4:
                break
        T = np.array(rows)
        return T[:3, :3], T[:3, 3]

    with open(path) as f:
        lines = f.readlines()

    ric, tic = np.eye(3), np.zeros(3)
    ric1, tic1 = np.eye(3), np.zeros(3)
    for i, line in enumerate(lines):
        if 'cam0:' in line:
            for j in range(i, min(i + 10, len(lines))):
                if 'T_cam_imu:' in lines[j]:
                    ric, tic = read_tf(lines, j + 1)
                    break
        if 'cam1:' in line:
            for j in range(i, min(i + 10, len(lines))):
                if 'T_cam_imu:' in lines[j]:
                    ric1, tic1 = read_tf(lines, j + 1)
                    break
    return ric, tic, ric1, tic1


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else '/tmp/vio_output'
    config = sys.argv[2] if len(sys.argv) > 2 else '/home/adrewn/Tassel/config/stereo_vins.yaml'

    print(f"Reading dump from: {prefix}*")
    print(f"Config: {config}")

    imu_ts, acc, gyro = read_imu(f'{prefix}_imu.txt')
    cam_ts, Rs, Ps, Vs, Bas, Bgs = read_state(f'{prefix}_state.txt')
    features_by_frame = read_features(f'{prefix}_features.txt')

    print(f"  IMU: {len(imu_ts)} measurements, {imu_ts[0]:.3f} -> {imu_ts[-1]:.3f} s")
    print(f"  State: {len(cam_ts)} frames, {cam_ts[0]:.3f} -> {cam_ts[-1]:.3f} s")
    print(f"  Features: {sum(len(v) for v in features_by_frame.values())} obs "
          f"across {len(features_by_frame)} frames")

    K = parse_yaml_k(config)
    ric, tic, ric1, tic1 = parse_yaml_extrinsics(config)
    baseline = norm(tic1 - tic)
    print(f"  K: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"  Baseline: {baseline:.4f} m")

    imu_dt = np.mean(np.diff(imu_ts))
    dt_step = (2.0 / 3.0) / (1.0 / imu_dt) if imu_dt > 0 else 0.0025
    print(f"  IMU dt: {imu_dt*1000:.2f} ms, integration step: {dt_step*1000:.3f} ms")

    obs_win = build_observations(features_by_frame, K, baseline)
    n_nonempty = sum(1 for o in obs_win if len(o) > 0)
    feat_frame_counts = defaultdict(list)
    for wi, obs in enumerate(obs_win):
        for lm_id, p_C, uv in obs:
            feat_frame_counts[lm_id].append(wi)
    n_multi = sum(1 for v in feat_frame_counts.values() if len(v) >= 2)
    print(f"  Observation frames: {n_nonempty} non-empty, "
          f"{n_multi} features seen in >=2 frames")

    if n_multi < 3:
        print("\n  WARNING: Too few multi-frame feature tracks.")
        return

    solver = RealDataSolver(imu_ts, gyro, acc, cam_ts, Rs, Ps, Vs, Bas, Bgs,
                            ric, tic, K, dt_step)

    p_opt, td_opt, rms, td_hist = solver.solve(obs_win, 0.0, fix_td=False)
    edges = build_edges(obs_win)

    print(f"\n  td estimated: {td_opt*1000:.2f} ms")
    print(f"  Edges: {len(edges)}")
    print(f"  Iterations: {len(td_hist)-1}")
    print(f"  td history: {[f'{t*1000:.1f}' for t in td_hist]}")

    # ── compute per-edge reprojection errors (before and after td correction) ──
    rerr_before = []  # td=0
    rerr_after = []   # td=td_opt
    imu_poses0, states0 = solver._full_state(0.0)
    imu_poses_opt, states_opt = solver._full_state(td_opt)
    for (i, j, px, py, pz, u_obs, v_obs) in edges:
        i, j = int(i), int(j)
        p_C_i = np.array([px, py, pz])
        uv_obs = np.array([u_obs, v_obs])
        for poses, err_list in [(imu_poses0, rerr_before), (imu_poses_opt, rerr_after)]:
            if poses is None:
                continue
            Q_wi = exp_so3(poses[i, :3])
            t_wi = poses[i, 3:]
            Q_wj = exp_so3(poses[j, :3])
            t_wj = poses[j, 3:]
            p_I = ric @ p_C_i + tic
            p_G = Q_wi @ p_I + t_wi
            p_I_j = Q_wj.T @ (p_G - t_wj)
            p_C_j = ric.T @ (p_I_j - tic)
            z = p_C_j[2]
            if z > 0.01:
                uv_pred = np.array([K[0, 0] * p_C_j[0] / z + K[0, 2],
                                    K[1, 1] * p_C_j[1] / z + K[1, 2]])
                err_list.append(norm(uv_obs - uv_pred))

    rms_before = np.sqrt(np.mean(np.array(rerr_before)**2)) if rerr_before else float('nan')
    rms_after = np.sqrt(np.mean(np.array(rerr_after)**2)) if rerr_after else float('nan')
    print(f"  RMS before td correction (td=0): {rms_before:.2f} px")
    print(f"  RMS after td correction (td={td_opt*1000:.2f} ms): {rms_after:.2f} px")

    # ── Figure: td convergence ──
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(len(td_hist)), np.array(td_hist) * 1000, 'b.-', lw=1.0, markersize=4)
    ax.axhline(td_opt * 1000, color='gray', ls='--', lw=0.7,
               label=f'estimated = {td_opt*1000:.2f} ms')
    ax.set_xlabel('Iteration'); ax.set_ylabel('$t_d$ (ms)')
    ax.set_title('$t_d$ convergence (real data, C++ dump)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}_td_estimation.png")
    print(f"Saved: {prefix}_td_estimation.png")
    plt.close(fig)

    # ── Figure: reprojection error histogram (before vs after td correction) ──
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if rerr_before:
        ax.hist(rerr_before, bins=30, color='#d7191c', alpha=0.5, edgecolor='white', lw=0.3,
                label=f'$t_d = 0$ ms (RMS = {rms_before:.2f} px)')
    if rerr_after:
        ax.hist(rerr_after, bins=30, color='#2b83ba', alpha=0.5, edgecolor='white', lw=0.3,
                label=f'$t_d = {td_opt*1000:.2f}$ ms (RMS = {rms_after:.2f} px)')
    ax.set_xlabel('Reprojection error (px)')
    ax.set_ylabel('Count')
    ax.set_title('Reprojection error distribution (Python re-analysis, OAK-D data)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{prefix}_reproj_error_hist.png")
    print(f"Saved: {prefix}_reproj_error_hist.png")
    plt.close(fig)

    # ── Figure: simulation vs real comparison (summary table as bar chart) ──
    sim_td = [-11.10, 4.0, 15.0, 23.56, 30.0]
    sim_est = [-11.20, 3.50, 14.20, 21.60, 29.20]
    sim_rms = [1.54, 1.17, 1.96, 3.09, 3.36]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    # td comparison
    x_sim = np.arange(len(sim_td))
    ax1.plot(sim_td, sim_est, 'go', label='Simulation', markersize=6)
    ax1.plot([td_opt * 1000], [td_opt * 1000], 'r*', markersize=12, label=f'Real (OAK-D)')
    ax1.plot([-30, 35], [-30, 35], 'k--', lw=0.5, label='$y=x$')
    ax1.set_xlabel('True / C++ $t_d$ (ms)')
    ax1.set_ylabel('Estimated $t_d$ (ms)')
    ax1.set_title('$t_d$: true vs estimated')
    ax1.legend(fontsize=7)
    # RMS comparison
    ax2.bar(x_sim, sim_rms, color='#abd9e9', label='Simulation (after $t_d$)')
    ax2.bar([len(sim_td)], [rms_after], color='#2b83ba', label='Real after $t_d$ (Python)')
    ax2.bar([len(sim_td) + 0.7], [rms_before], color='#d7191c', label='Real before $t_d$ (Python)')
    ax2.set_xticks(list(x_sim) + [len(sim_td)])
    ax2.set_xticklabels([f'{t:.0f}' for t in sim_td] + [f'Real\n(C++)'])
    ax2.set_ylabel('RMS (px)')
    ax2.set_title('Reprojection RMS')
    ax2.legend(fontsize=7)
    fig.suptitle('Simulation vs Real Hardware Validation')
    fig.tight_layout()
    fig.savefig(f"{prefix}_sim_vs_real.png")
    print(f"Saved: {prefix}_sim_vs_real.png")
    plt.close(fig)

    # ── Figure: point cloud comparison (td=0 vs td=td_opt) ──
    # Triangulate landmarks in world frame using host frame's stereo depth
    # and host frame's IMU pose at td=0 vs td=td_opt
    p_G_before = []  # td = 0
    p_G_after = []   # td = td_opt
    for (i, j, px, py, pz, u_obs, v_obs) in edges:
        i = int(i)
        p_C_i = np.array([px, py, pz])
        for poses, p_list in [(imu_poses0, p_G_before), (imu_poses_opt, p_G_after)]:
            if poses is None:
                continue
            Q_wi = exp_so3(poses[i, :3])
            t_wi = poses[i, 3:]
            p_I = ric @ p_C_i + tic
            p_G = Q_wi @ p_I + t_wi
            p_list.append(p_G)

    fig = plt.figure(figsize=(12, 5))
    for idx, (pts, label, color) in enumerate([
        (p_G_before, '$t_d = 0$ ms (uncompensated)', '#d7191c'),
        (p_G_after, f'$t_d = {td_opt*1000:.2f}$ ms (compensated)', '#2b83ba')]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        pts = np.array(pts)
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=3, alpha=0.6)
        if idx == 0:
            traj = imu_poses0
        else:
            traj = imu_poses_opt
        if traj is not None:
            ax.plot(traj[:, 3], traj[:, 4], traj[:, 5], 'k.-', lw=0.8, markersize=3, alpha=0.5,
                    label='camera path')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(label)
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('Landmark point cloud: $t_d$ compensation comparison (Python re-analysis)')
    fig.tight_layout()
    fig.savefig(f"{prefix}_pointcloud_compare.png", dpi=300)
    print(f"Saved: {prefix}_pointcloud_compare.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
