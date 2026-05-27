import numpy as np
from numpy.linalg import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── publication style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'lines.linewidth': 1.0, 'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'text.usetex': False,
})

# ── geometry utils ─────────────────────────────────────────────────
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def exp_so3(phi):
    theta = norm(phi)
    if theta < 1e-12:
        return np.eye(3)
    a = phi / theta
    return np.eye(3) + np.sin(theta) * skew(a) + (1 - np.cos(theta)) * skew(a) @ skew(a)

def camera_pose(R_imu, P_imu, ric, tic):
    return R_imu @ ric, P_imu + R_imu @ tic

def project(p_G, R_GI, P_GI, ric, tic, K):
    p_I = R_GI.T @ (p_G - P_GI)
    p_C = ric.T @ (p_I - tic)
    z = p_C[2]
    uv = p_C[:2] / z
    px = K[:2, :2] @ uv + K[:2, 2]
    return px, p_C

def imu_pose_at(imu_ts, R_gt, v_gt, P_gt, gyro_body, acc_body, t):
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
    gyro_t = (1 - alpha) * gyro_body[i0] + alpha * gyro_body[i1]
    acc_t = (1 - alpha) * acc_body[i0] + alpha * acc_body[i1]
    gyro_mid = 0.5 * (gyro_body[i0] + gyro_t)
    acc_mid = 0.5 * (acc_body[i0] + acc_t)
    dR = exp_so3(gyro_mid * dt)
    R = R_gt[i0] @ dR
    v = v_gt[i0] + R_gt[i0] @ acc_mid * dt
    P = P_gt[i0] + v_gt[i0] * dt + 0.5 * R_gt[i0] @ acc_mid * dt**2
    return R, v, P, gyro_t, acc_t

def compute_td_jacobian(K, p_C, ric, R_GI, P_GI, p_G, v_G, w_G_body, a_body, dt):
    x, y, z = p_C[0], p_C[1], p_C[2]
    fx, fy = K[0, 0], K[1, 1]
    reduce = np.array([[fx/z, 0, -fx*x/(z*z)],
                       [0, fy/z, -fy*y/(z*z)]])
    w_G = R_GI @ w_G_body
    p_I = R_GI.T @ (p_G - P_GI)
    exp_w_dt = exp_so3(w_G_body * dt)
    inner = (skew(w_G) @ p_I - R_GI.T @ v_G - (exp_w_dt @ a_body + a_body) * dt)
    return reduce @ ric.T @ inner

# ── simulation ─────────────────────────────────────────────────────
def simulate(t_d_true, imu_freq, cam_freq, duration):
    dt_imu = 1.0 / imu_freq
    n_imu = int(duration / dt_imu)
    imu_ts = np.arange(n_imu) * dt_imu

    gyro_body = np.zeros((n_imu, 3))
    acc_body = np.zeros((n_imu, 3))
    for i in range(n_imu):
        t = imu_ts[i]
        gyro_body[i] = np.array([
            2.0 * np.sin(2*np.pi*0.6*t),
            1.5 * np.cos(2*np.pi*0.9*t + 1.0),
            1.0 * np.sin(2*np.pi*1.3*t + 0.3)])
        acc_body[i] = np.array([
            3.0 * np.cos(2*np.pi*0.4*t + 0.7),
            2.0 * np.sin(2*np.pi*0.55*t),
            1.0 * np.cos(1.8*np.pi*t) + 9.81])

    R_gt = np.zeros((n_imu, 3, 3)); v_gt = np.zeros((n_imu, 3)); P_gt = np.zeros((n_imu, 3))
    R_gt[0] = np.eye(3)
    for i in range(n_imu - 1):
        gm = 0.5 * (gyro_body[i] + gyro_body[i+1])
        am = 0.5 * (acc_body[i] + acc_body[i+1])
        R_gt[i+1] = R_gt[i] @ exp_so3(gm * dt_imu)
        v_gt[i+1] = v_gt[i] + R_gt[i] @ am * dt_imu
        P_gt[i+1] = P_gt[i] + v_gt[i] * dt_imu + 0.5 * R_gt[i] @ am * dt_imu**2

    ric = exp_so3(np.array([0.02, -0.01, 0.03]))
    tic = np.array([0.05, 0.01, -0.02])
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]])

    # generate landmarks visible from multiple camera poses along trajectory
    np.random.seed(42)
    n_landmarks = 40
    landmarks = []
    # sample camera poses spaced across the trajectory
    sample_ts = np.linspace(0.5, duration - 0.5, 10)
    attempts = 0
    while len(landmarks) < n_landmarks and attempts < 500:
        attempts += 1
        # pick a random camera pose from the trajectory
        t_cam = np.random.choice(sample_ts)
        idx = np.searchsorted(imu_ts, t_cam)
        Rc, Pc = camera_pose(R_gt[idx], P_gt[idx], ric, tic)
        # random point in camera frustum at this pose
        depth = np.random.uniform(1.5, 15.0)
        u = np.random.uniform(60, 580)
        v = np.random.uniform(40, 440)
        p_C = np.array([(u - K[0, 2]) * depth / K[0, 0],
                        (v - K[1, 2]) * depth / K[1, 1],
                        depth])
        p_G = Rc @ p_C + Pc
        # accept if visible from at least 3 of the sampled poses
        vis_count = 0
        for ts in sample_ts:
            idx_s = np.searchsorted(imu_ts, ts)
            R_s, _, P_s, _, _ = imu_pose_at(imu_ts, R_gt, v_gt, P_gt, gyro_body, acc_body, ts)
            px, _ = project(p_G, R_s, P_s, ric, tic, K)
            if 15 < px[0] < 625 and 15 < px[1] < 465:
                vis_count += 1
        if vis_count >= 3:
            landmarks.append(p_G)
    n_landmarks = len(landmarks)

    cam_ts = np.arange(0.5, duration - 0.1, 1.0 / cam_freq)

    t_d_est = 0.0
    history = [t_d_est]

    # record all camera poses for trajectory comparison
    cam_pos_wrong = np.zeros((len(cam_ts), 3))
    cam_pos_correct = np.zeros((len(cam_ts), 3))
    cam_pos_est = np.zeros((len(cam_ts), 3))
    cam_rot_wrong = np.zeros((len(cam_ts), 3, 3))
    cam_rot_correct = np.zeros((len(cam_ts), 3, 3))
    cam_rot_est = np.zeros((len(cam_ts), 3, 3))
    reproj_errors = []

    record_indices = [0, len(cam_ts)//3, 2*len(cam_ts)//3, len(cam_ts)-1]
    sampled_wrong, sampled_correct, sampled_est = [], [], []

    for k, t_cam in enumerate(cam_ts):
        t_true = max(t_cam + t_d_true, imu_ts[0])
        R_true, _, P_true, _, _ = imu_pose_at(imu_ts, R_gt, v_gt, P_gt, gyro_body, acc_body, t_true)

        R_wrong, _, P_wrong, _, _ = imu_pose_at(imu_ts, R_gt, v_gt, P_gt, gyro_body, acc_body, t_cam)

        # joint estimation from all landmarks in this single frame
        t_est = max(t_cam + t_d_est, imu_ts[0])
        R_est, v_est, P_est, w_body, a_body = imu_pose_at(
            imu_ts, R_gt, v_gt, P_gt, gyro_body, acc_body, t_est)

        H_joint = 0.0
        b_joint = 0.0
        r_wrong_sum = 0.0
        r_corrected_sum = 0.0
        n_visible = 0

        for p_G in landmarks:
            px_meas, _ = project(p_G, R_true, P_true, ric, tic, K)
            if px_meas[0] < 5 or px_meas[0] > 635 or px_meas[1] < 5 or px_meas[1] > 475:
                continue
            n_visible += 1

            px_pred, p_C = project(p_G, R_est, P_est, ric, tic, K)
            r = px_meas - px_pred

            px_wrong, _ = project(p_G, R_wrong, P_wrong, ric, tic, K)
            r_w = px_meas - px_wrong

            J = compute_td_jacobian(K, p_C, ric, R_est, P_est, p_G, v_est, w_body, a_body, dt_imu)
            H_joint += float(J.T @ J)
            b_joint += float(J.T @ r)
            r_wrong_sum += float(r_w.T @ r_w)
            r_corrected_sum += float(r.T @ r)

        if n_visible > 0 and H_joint > 1e-12:
            lm_lambda = 1e6  # fixed damping: damps low-H frames, trusts high-H
            delta = -b_joint / (H_joint + lm_lambda)
            delta = np.clip(delta, -2.0 * dt_imu, 2.0 * dt_imu)
            t_d_est += delta

        t_d_est = max(t_d_est, -0.1)
        history.append(t_d_est)

        # store full camera pose
        Rc_w, Pc_w = camera_pose(R_wrong, P_wrong, ric, tic)
        Rc_c, Pc_c = camera_pose(R_true, P_true, ric, tic)
        Rc_e, Pc_e = camera_pose(R_est, P_est, ric, tic)
        cam_pos_wrong[k] = Pc_w
        cam_pos_correct[k] = Pc_c
        cam_pos_est[k] = Pc_e
        cam_rot_wrong[k] = Rc_w
        cam_rot_correct[k] = Rc_c
        cam_rot_est[k] = Rc_e

        reproj_errors.append((k, np.sqrt(r_wrong_sum / max(n_visible, 1)),
                              np.sqrt(r_corrected_sum / max(n_visible, 1))))

        if k in record_indices:
            sampled_wrong.append(Pc_w)
            sampled_correct.append(Pc_c)
            sampled_est.append(Pc_e)

    return {
        'history': np.array(history),
        'cam_ts': cam_ts,
        'imu_ts': imu_ts,
        'P_gt': P_gt, 'R_gt': R_gt, 'v_gt': v_gt,
        'gyro': gyro_body, 'acc': acc_body,
        'ric': ric, 'tic': tic, 'K': K,
        'cam_pos_wrong': cam_pos_wrong,
        'cam_pos_correct': cam_pos_correct,
        'cam_pos_est': cam_pos_est,
        'cam_rot_wrong': cam_rot_wrong,
        'cam_rot_correct': cam_rot_correct,
        'cam_rot_est': cam_rot_est,
        'sampled_wrong': sampled_wrong,
        'sampled_correct': sampled_correct,
        'sampled_est': sampled_est,
        'record_indices': record_indices,
        'reproj_errors': reproj_errors,
    }

# ── main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    true_delays = [0.004, 0.015, 0.030]
    sim_results = {}
    for td in true_delays:
        sim_results[td] = simulate(td, imu_freq=400, cam_freq=30, duration=5.0)
        h = sim_results[td]['history']
        print(f"t_d_true={td*1000:.0f}ms: final={h[-1]*1000:.2f}ms, "
              f"err={abs(h[-1]-td)*1000:.2f}ms, frames={len(h)-1}")

    # ═══════════════════════════════════════════════════════════════
    # Figure 1: IMU excitation signals
    # ═══════════════════════════════════════════════════════════════
    r = sim_results[0.015]
    imu_ts, gyro, acc = r['imu_ts'], r['gyro'], r['acc']
    fig1, axes1 = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    t_plot = imu_ts[:800]
    for ax, data, ylabel, title in [
        (axes1[0], gyro[:800], 'Angular velocity (rad/s)', 'IMU Gyroscope'),
        (axes1[1], acc[:800], 'Acceleration (m/s$^2$)', 'IMU Accelerometer')]:
        ax.plot(t_plot, data[:, 0], label='x', lw=0.8)
        ax.plot(t_plot, data[:, 1], label='y', lw=0.8)
        ax.plot(t_plot, data[:, 2], label='z', lw=0.8)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc='upper right', ncol=3)
    axes1[1].set_xlabel('Time (s)')
    fig1.tight_layout()
    fig1.savefig("/home/adrewn/Tassel/scripts/time_delay_imu_signals.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 2: t_d convergence
    # ═══════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    colors = ['#2166ac', '#b2182b', '#4daf4a']
    for idx, td in enumerate(true_delays):
        h = sim_results[td]['history']
        ax2.plot(np.arange(len(h)), h * 1000, color=colors[idx], lw=1.2,
                 label=f'true = {td*1000:.0f} ms')
        ax2.axhline(y=td * 1000, color=colors[idx], linestyle='--', lw=0.7, alpha=0.6)
        ax2.annotate(f'{h[-1]*1000:.2f} ms', xy=(len(h)-1, h[-1]*1000),
                     xytext=(len(h)+3, h[-1]*1000 + 1.5), fontsize=7,
                     color=colors[idx],
                     arrowprops=dict(arrowstyle='->', color=colors[idx], lw=0.5))
    ax2.set_xlabel('Frame index'); ax2.set_ylabel('Estimated $t_d$ (ms)')
    ax2.set_title('Time delay convergence')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("/home/adrewn/Tassel/scripts/time_delay_convergence.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 3: Reprojection error
    # ═══════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for idx, td in enumerate(true_delays):
        rerr = np.array(sim_results[td]['reproj_errors'])
        axes3[idx].plot(rerr[:, 0], rerr[:, 1], color='#d7191c', lw=0.8,
                        label='Before ($t_d=0$)')
        axes3[idx].plot(rerr[:, 0], rerr[:, 2], color='#2b83ba', lw=0.8,
                        label='After ($t_d$ estimated)')
        axes3[idx].set_title(f'$t_d^{{true}} = {td*1000:.0f}$ ms')
        axes3[idx].set_xlabel('Frame index')
        if idx == 0:
            axes3[idx].set_ylabel('Reprojection error (px)')
        axes3[idx].legend(fontsize=7)
    fig3.tight_layout()
    fig3.savefig("/home/adrewn/Tassel/scripts/time_delay_reproj_error.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 4: 3D trajectory + sampled camera poses
    # ═══════════════════════════════════════════════════════════════
    fig4 = plt.figure(figsize=(14, 5))
    for idx, td in enumerate(true_delays):
        r = sim_results[td]
        P_gt = r['P_gt']
        cw = np.array(r['sampled_wrong'])
        cc = np.array(r['sampled_correct'])
        ce = np.array(r['sampled_est'])

        ax = fig4.add_subplot(1, 3, idx + 1, projection='3d')
        ax.plot(P_gt[::40, 0], P_gt[::40, 1], P_gt[::40, 2],
                color='gray', alpha=0.25, lw=0.4)
        ax.scatter(cw[:, 0], cw[:, 1], cw[:, 2], c='#d7191c', s=25, marker='x',
                   label='Wrong ($t_d=0$)', zorder=5)
        ax.scatter(cc[:, 0], cc[:, 1], cc[:, 2], c='#2b83ba', s=25, marker='o',
                   label='Correct (true $t_d$)', zorder=5)
        ax.scatter(ce[:, 0], ce[:, 1], ce[:, 2], c='#4daf4a', s=18, marker='^',
                   label='Estimated $t_d$', zorder=5)
        for i in range(len(cw)):
            ax.plot([cw[i, 0], cc[i, 0]], [cw[i, 1], cc[i, 1]], [cw[i, 2], cc[i, 2]],
                    color='#fdae61', lw=0.5, alpha=0.6)
        ax.scatter(*cw[0], c='#d7191c', s=50, marker='x', zorder=6)
        ax.set_title(f'$t_d^{{true}} = {td*1000:.0f}$ ms')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.legend(fontsize=6, loc='upper left')
        pts = np.vstack([cw, cc, P_gt[::40]])
        ranges = pts.max(axis=0) - pts.min(axis=0)
        mid = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
        rng = 0.6 * ranges.max()
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)
    fig4.tight_layout()
    fig4.savefig("/home/adrewn/Tassel/scripts/time_delay_3d_poses.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 5: Camera pose triad comparison (30 ms case, 3 keyframes)
    # ═══════════════════════════════════════════════════════════════
    r30 = sim_results[0.030]
    P_gt30 = r30['P_gt']; cam_ts30 = r30['cam_ts']
    pos_w = r30['cam_pos_wrong']; pos_c = r30['cam_pos_correct']
    rot_w = r30['cam_rot_wrong']; rot_c = r30['cam_rot_correct']

    n_cam = len(cam_ts30)
    key_frames = [n_cam//4, n_cam//2, 3*n_cam//4]
    axis_len = 0.20
    axis_colors = ['#e41a1c', '#4daf4a', '#377eb8']  # R,G,B for X,Y,Z

    def draw_camera_axes(ax, pos, rot, alpha_val, scale=1.0):
        """Draw camera triad as line segments (cleaner than quiver)."""
        al = axis_len * scale
        for d, c in enumerate(axis_colors):
            end = pos + rot[:, d] * al
            ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                    color=c, lw=1.5 * scale, alpha=alpha_val)

    fig5 = plt.figure(figsize=(14, 5))
    for sub_idx, kf in enumerate(key_frames):
        ax = fig5.add_subplot(1, 3, sub_idx + 1, projection='3d')

        # local IMU segment around this frame
        t_cam_kf = cam_ts30[kf]
        t_win = 0.3
        mask = (r30['imu_ts'] >= t_cam_kf - t_win) & (r30['imu_ts'] <= t_cam_kf + t_win)
        t_local = r30['imu_ts'][mask]
        P_local = P_gt30[mask]
        ax.plot(P_local[::5, 0], P_local[::5, 1], P_local[::5, 2],
                color='gray', alpha=0.25, lw=0.5)

        # wrong camera (t_d = 0)
        draw_camera_axes(ax, pos_w[kf], rot_w[kf], 0.55, scale=1.0)
        ax.scatter(*pos_w[kf], c='#d7191c', s=40, marker='x', zorder=5)

        # correct camera (true t_d)
        draw_camera_axes(ax, pos_c[kf], rot_c[kf], 0.9, scale=1.0)
        ax.scatter(*pos_c[kf], c='#2b83ba', s=40, marker='o', zorder=5)

        # displacement line
        ax.plot([pos_w[kf][0], pos_c[kf][0]],
                [pos_w[kf][1], pos_c[kf][1]],
                [pos_w[kf][2], pos_c[kf][2]],
                color='#fdae61', lw=1.0, alpha=0.7, linestyle='--')

        dist = norm(pos_w[kf] - pos_c[kf]) * 100
        ax.set_title(f'Frame {kf} ($t_{{cam}}$={t_cam_kf:.2f}s, '
                     f'$\\|\\Delta P\\|$={dist:.1f}cm)')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

        # zoom
        center = 0.5 * (pos_w[kf] + pos_c[kf])
        rng = 0.5
        ax.set_xlim(center[0] - rng, center[0] + rng)
        ax.set_ylim(center[1] - rng, center[1] + rng)
        ax.set_zlim(center[2] - rng, center[2] + rng)

    # legend as text
    fig5.text(0.12, 0.02, 'Red=X  Green=Y  Blue=Z  |  '
              '— Wrong ($t_d$=0)  — Correct (true $t_d$)  -- Displacement',
              fontsize=7, ha='left', transform=fig5.transFigure)
    fig5.tight_layout(rect=[0, 0.04, 1, 1])
    fig5.savefig("/home/adrewn/Tassel/scripts/time_delay_camera_axes.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 6: Position error vs frame
    # ═══════════════════════════════════════════════════════════════
    fig6, axes6 = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for idx, td in enumerate(true_delays):
        r = sim_results[td]
        pos_err_before = norm(r['cam_pos_wrong'] - r['cam_pos_correct'], axis=1) * 100  # cm
        pos_err_after = norm(r['cam_pos_est'] - r['cam_pos_correct'], axis=1) * 100
        frames = np.arange(len(pos_err_before))
        axes6[idx].plot(frames, pos_err_before, color='#d7191c', lw=0.8,
                        label='Before ($t_d=0$)')
        axes6[idx].plot(frames, pos_err_after, color='#2b83ba', lw=0.8,
                        label='After ($t_d$ estimated)')
        axes6[idx].set_title(f'$t_d^{{true}} = {td*1000:.0f}$ ms')
        axes6[idx].set_xlabel('Frame index')
        if idx == 0:
            axes6[idx].set_ylabel('Position error (cm)')
        axes6[idx].legend(fontsize=7)
    fig6.tight_layout()
    fig6.savefig("/home/adrewn/Tassel/scripts/time_delay_position_error.png")

    print("\nSaved:")
    print("  scripts/time_delay_imu_signals.png")
    print("  scripts/time_delay_convergence.png")
    print("  scripts/time_delay_reproj_error.png")
    print("  scripts/time_delay_3d_poses.png")
    print("  scripts/time_delay_camera_axes.png")
    print("  scripts/time_delay_position_error.png")
