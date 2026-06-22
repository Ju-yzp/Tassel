#!/usr/bin/env python3
"""Static multi-pose IMU calibration for OAK cameras via Bundle Adjustment.

Place the device stationary in 6+ distinct orientations.
Jointly optimizes accelerometer bias, scale, and per-pose gravity directions
by minimizing the full 3D residual: a_corrected = S*(a_raw - b) = g_i, |g_i| = g_norm.

Gyroscope bias is the mean of all stationary readings.

Usage:
    python oak_imu_calib.py [--poses N] [--duration S] [--output FILE]
"""

import argparse
import time
import sys

import depthai as dai
import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def is_stationary(acc_buffer, gyro_buffer, acc_var_thres=0.005, gyro_var_thres=0.002):
    if len(acc_buffer) < 10:
        return False
    acc_var = np.var(np.linalg.norm(acc_buffer, axis=1))
    gyro_var = np.var(np.linalg.norm(gyro_buffer, axis=1))
    return acc_var < acc_var_thres and gyro_var < gyro_var_thres


def collect_static_sample(pipeline, duration=3.0, settle_time=3.0):
    with dai.Device(pipeline) as device:
        imu_queue = device.getOutputQueue(name="imu", maxSize=100, blocking=False)

        acc_window = []
        gyr_window = []
        acc_samples = []
        gyr_samples = []

        print(f"    Settling ({settle_time:.0f}s)...", end="", flush=True)
        settle_start = time.monotonic()
        last_status = time.monotonic()

        while True:
            imu_data = imu_queue.tryGet()
            if imu_data is None:
                time.sleep(0.005)
                continue

            for pkt in imu_data.packets:
                acc = np.array([pkt.acceleroMeter.x, pkt.acceleroMeter.y, pkt.acceleroMeter.z])
                gyr = np.array([pkt.gyroscope.x, pkt.gyroscope.y, pkt.gyroscope.z])
                acc_window.append(acc)
                gyr_window.append(gyr)
                if len(acc_window) > 50:
                    acc_window.pop(0)
                    gyr_window.pop(0)

            now = time.monotonic()
            if now - settle_start < settle_time:
                continue

            if is_stationary(acc_window, gyr_window):
                if now - last_status > 0.5:
                    print(".", end="", flush=True)
                    last_status = now
                acc_samples.extend(acc_window)
                gyr_samples.extend(gyr_window)
                acc_window.clear()
                gyr_window.clear()

                if len(acc_samples) >= 100:
                    acc_mean = np.mean(acc_samples, axis=0)
                    gyr_mean = np.mean(gyr_samples, axis=0)
                    acc_std = np.std(acc_samples, axis=0)
                    raw_norm = np.linalg.norm(acc_mean)
                    if abs(raw_norm - 9.80) > 0.5:
                        print(f" WARN: |a_raw|={raw_norm:.2f} far from g, may not be stationary!")
                    else:
                        print(" done")
                    return acc_mean, gyr_mean, acc_std

            if now - settle_start > settle_time + duration + 5.0:
                print(" timeout!")
                return None, None, None


def create_pipeline():
    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)
    xlink_out = pipeline.create(dai.node.XLinkOut)
    xlink_out.setStreamName("imu")

    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(1)

    imu.out.link(xlink_out.input)
    return pipeline


# ---------------------------------------------------------------------------
# BA calibration
# ---------------------------------------------------------------------------

def spherical_to_vector(theta, phi, r=1.0):
    """Spherical coordinates to 3D vector. theta: polar [0,pi], phi: azimuth [0,2pi]."""
    return r * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def calibrate_accel_ba(poses_acc, g_norm=9.8035, full_scale=False):
    """Bundle adjustment for IMU accelerometer calibration.

    Parameters
    ----------
    poses_acc : list of (3,) ndarray
        Raw accelerometer readings in each static pose.
    g_norm : float
        Known gravity magnitude.
    full_scale : bool
        If True, use full 3x3 scale matrix (9 scale params + 3 bias).
        If False, use diagonal scale (3 scale params + 3 bias).

    Returns
    -------
    bias : (3,) ndarray
    scale : (3,) ndarray or (3,3) ndarray
    g_dirs : list of (3,) ndarray, estimated gravity directions per pose.
    """
    n = len(poses_acc)

    if full_scale:
        # Parameters: bias(3) + scale(9) + theta(phi) per pose(2n) = 12 + 2n
        n_param = 12 + 2 * n

        def unpack(x):
            b = x[:3]
            S = x[3:12].reshape(3, 3)
            theta_phi = x[12:].reshape(n, 2)
            return b, S, theta_phi

        def residuals(x):
            b, S, theta_phi = unpack(x)
            res = []
            for i in range(n):
                theta, phi = theta_phi[i]
                g_i = spherical_to_vector(theta, phi, g_norm)
                a_corr = S @ (poses_acc[i] - b)
                res.extend(a_corr - g_i)
            return np.array(res)

        x0 = np.zeros(n_param)
        x0[3:12] = np.eye(3).ravel()  # scale init = I
        # Initialize theta, phi from raw readings
        for i in range(n):
            a = poses_acc[i]
            a_norm = np.linalg.norm(a)
            if a_norm > 1e-6:
                theta = np.arccos(np.clip(a[2] / a_norm, -1, 1))
                phi = np.arctan2(a[1], a[0])
            else:
                theta, phi = 0.0, 0.0
            x0[12 + 2 * i] = theta
            x0[12 + 2 * i + 1] = phi

    else:
        # Diagonal scale: bias(3) + scale(3) + theta,phi per pose(2n) = 6 + 2n
        n_param = 6 + 2 * n

        def unpack(x):
            b = x[:3]
            S = x[3:6]
            theta_phi = x[6:].reshape(n, 2)
            return b, S, theta_phi

        def residuals(x):
            b, S, theta_phi = unpack(x)
            res = []
            for i in range(n):
                theta, phi = theta_phi[i]
                g_i = spherical_to_vector(theta, phi, g_norm)
                a_corr = S * (poses_acc[i] - b)
                res.extend(a_corr - g_i)
            return np.array(res)

        x0 = np.zeros(n_param)
        x0[3:6] = 1.0  # scale init
        for i in range(n):
            a = poses_acc[i]
            a_norm = np.linalg.norm(a)
            if a_norm > 1e-6:
                theta = np.arccos(np.clip(a[2] / a_norm, -1, 1))
                phi = np.arctan2(a[1], a[0])
            else:
                theta, phi = 0.0, 0.0
            x0[6 + 2 * i] = theta
            x0[6 + 2 * i + 1] = phi

    result = least_squares(residuals, x0, method="lm", verbose=0,
                           xtol=1e-8, ftol=1e-8)

    if full_scale:
        b, S, theta_phi = unpack(result.x)
        scale = S
    else:
        b, S, theta_phi = unpack(result.x)
        scale = S

    g_dirs = []
    for i in range(n):
        theta, phi = theta_phi[i]
        g_dirs.append(spherical_to_vector(theta, phi, g_norm))

    return b, scale, g_dirs


def calibrate_accel_bias_only(poses_acc, g_norm=9.8035):
    """Bias-only LM: residual = |a - b| - g_norm. 3 parameters, no scale coupling."""
    x0 = np.zeros(3)

    def residuals(x):
        b = x
        res = []
        for a in poses_acc:
            res.append(np.linalg.norm(a - b) - g_norm)
        return np.array(res)

    result = least_squares(residuals, x0, method="lm", xtol=1e-8, ftol=1e-8)
    return result.x


def calibrate_accel_norm_only(poses_acc, g_norm=9.8035):
    """|a|=g calibration with bias+diag scale (6 params). For comparison only."""
    x0 = np.zeros(6)
    x0[3:] = 1.0

    def residuals(x):
        bx, by, bz, sx, sy, sz = x
        res = []
        for a in poses_acc:
            a_corr = np.array([sx, sy, sz]) * (a - np.array([bx, by, bz]))
            res.append(np.linalg.norm(a_corr) - g_norm)
        return np.array(res)

    result = least_squares(residuals, x0, method="lm")
    return result.x[:3], result.x[3:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Static multi-pose OAK IMU calibration (BA)")
    parser.add_argument("--poses", type=int, default=10, help="Number of static poses (default: 10)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Recording duration per pose in seconds")
    parser.add_argument("--output", type=str, default=None, help="Output YAML file")
    parser.add_argument("--g-norm", type=float, default=9.8035,
                        help="Local gravity magnitude")
    parser.add_argument("--full-scale", action="store_true",
                        help="Use full 3x3 scale matrix (9 params) instead of diagonal (3 params)")
    args = parser.parse_args()

    scale_type = "full 3x3" if args.full_scale else "diagonal"
    print(f"OAK IMU Static Multi-Pose Calibration (BA)")
    print(f"  Scale: {scale_type} | Poses: {args.poses} | Duration: {args.duration}s")
    print(f"  g = {args.g_norm:.4f} m/s²")
    print()

    pipeline = create_pipeline()
    poses_acc = []
    poses_gyr = []

    for i in range(args.poses):
        print(f"--- Pose {i + 1}/{args.poses} ---")
        input("  Place device in new orientation, then press ENTER...")

        acc_mean, gyr_mean, acc_std = collect_static_sample(pipeline, duration=args.duration)
        if acc_mean is None:
            print("  Failed to stabilize! Re-do this pose.")
            continue

        poses_acc.append(acc_mean)
        poses_gyr.append(gyr_mean)
        print(
            f"  Acc: [{acc_mean[0]:+.4f}, {acc_mean[1]:+.4f}, {acc_mean[2]:+.4f}] "
            f"|a|={np.linalg.norm(acc_mean):.4f}"
        )

    # ---- Diagnostics ----
    raw_norms = [np.linalg.norm(a) for a in poses_acc]
    print(f"\n{'='*60}")
    print(f"Raw data")
    print(f"  |a| range: {min(raw_norms):.4f} ~ {max(raw_norms):.4f}  "
          f"(target {args.g_norm:.4f}, spread {(max(raw_norms)-min(raw_norms)):.4f})")

    # ---- Bias-only LM (primary) ----
    acc_bias = calibrate_accel_bias_only(poses_acc, args.g_norm)

    # ---- BA for comparison ----
    bias_ba, acc_scale, g_dirs = calibrate_accel_ba(
        poses_acc, g_norm=args.g_norm, full_scale=args.full_scale
    )

    # ---- Gyro ----
    all_gyr = np.array(poses_gyr)
    gyr_bias = np.mean(all_gyr, axis=0)

    # ---- Report ----
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")

    print(f"  Bias-only LM (|a-b|=g):")
    print(f"    bias = [{acc_bias[0]:+.6f}, {acc_bias[1]:+.6f}, {acc_bias[2]:+.6f}]")

    if args.full_scale:
        print(f"  BA  (full 3x3, comparison):")
        print(f"    bias = [{bias_ba[0]:+.6f}, {bias_ba[1]:+.6f}, {bias_ba[2]:+.6f}]")
        print(f"    S =")
        for row in acc_scale:
            print(f"        [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")
    else:
        print(f"  BA  (diag, comparison):")
        print(f"    bias = [{bias_ba[0]:+.6f}, {bias_ba[1]:+.6f}, {bias_ba[2]:+.6f}]")
        print(f"    S    = [{acc_scale[0]:+.6f}, {acc_scale[1]:+.6f}, {acc_scale[2]:+.6f}]")

    print(f"  Gyro bias: [{gyr_bias[0]:+.6f}, {gyr_bias[1]:+.6f}, {gyr_bias[2]:+.6f}]")

    # ---- Per-pose residual (bias-only) ----
    print(f"\n  Per-pose residual (bias-only, |a-b|-g):")
    for i in range(len(poses_acc)):
        a_corr = poses_acc[i] - acc_bias
        res = np.linalg.norm(a_corr) - args.g_norm
        print(f"    Pose {i+1}: |a-b|={np.linalg.norm(a_corr):.4f}  "
              f"res={res:+.4f}")

    # ---- Config output ----
    print(f"\n  For config YAML:")
    print(f"    acc_bias: {acc_bias.tolist()}")

    if args.output:
        import yaml
        out = {
            "acc_bias": acc_bias.tolist(),
            "gyr_bias": gyr_bias.tolist(),
            "g_norm": args.g_norm,
        }
        with open(args.output, "w") as f:
            yaml.dump(out, f, default_flow_style=None)
        print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
