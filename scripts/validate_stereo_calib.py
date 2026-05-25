#!/usr/bin/env python3
"""Real-time stereo calibration validation using AprilGrid re-projection error."""
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from collections import deque

# ── New calibration from stereo_vins.yaml ──
CAM0 = {
    "K": np.array([[455.792984, 0.0, 328.543398],
                   [0.0, 455.436223, 224.260803],
                   [0.0, 0.0, 1.0]]),
    "D": np.array([0.009724719, -0.000986831, -0.000359909, 0.000183854]),
}
CAM1 = {
    "K": np.array([[459.495739, 0.0, 316.330754],
                   [0.0, 459.271937, 229.892695],
                   [0.0, 0.0, 1.0]]),
    "D": np.array([0.021736064, -0.017403044, -0.000363312, -0.001028484]),
}

# ── AprilGrid 6x6, 3cm tag, 0.3 spacing ──
TAG_SIZE = 0.03
TAG_SPACING = 0.03 * 0.3
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
BOARD = cv2.aruco.GridBoard_create(6, 6, TAG_SIZE, TAG_SPACING, ARUCO_DICT)

# ── Rolling statistics ──
WINDOW = 100
err_left = deque(maxlen=WINDOW)
err_right = deque(maxlen=WINDOW)
count_left = deque(maxlen=WINDOW)
count_right = deque(maxlen=WINDOW)

bridge = CvBridge()


def detect_and_reproject(img, K, D, cam_name):
    """Detect AprilGrid, compute re-projection error via PnP."""
    corners, ids, _ = cv2.aruco.detectMarkers(img, ARUCO_DICT)
    if ids is None or len(ids) < 4:
        return None, 0, img

    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for c in corners:
        cv2.cornerSubPix(img, c, (5, 5), (-1, -1), criteria)

    # Get board 3D points and matching image points
    ret, obj_pts, img_pts = cv2.aruco.Board_matchImagePoints(
        BOARD, corners, ids, np.array([]), np.array([]))

    if ret < 4:
        return None, ret, img

    # Solve PnP with full model
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None, ret, img

    # Project 3D points back to image
    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)

    # Compute re-projection error (in distorted image plane)
    img_pts = img_pts.reshape(-1, 2)
    proj_pts = proj_pts.reshape(-1, 2)
    errors = np.linalg.norm(img_pts - proj_pts, axis=1)
    mean_err = errors.mean()
    max_err = errors.max()

    # Draw
    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    for i, (ip, pp) in enumerate(zip(img_pts, proj_pts)):
        cv2.line(img, tuple(ip.astype(int)), tuple(pp.astype(int)), (0, 0, 255), 1)
    cv2.putText(img, f"{cam_name} reproj: {mean_err:.2f}px (#{ret})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return mean_err, ret, img


def left_cb(msg):
    img = bridge.imgmsg_to_cv2(msg, "mono8")
    err, n, vis = detect_and_reproject(img, CAM0["K"], CAM0["D"], "L")
    if err is not None:
        err_left.append(err)
        count_left.append(n)

    # Print stats every 30 frames
    if len(err_left) > 0 and msg.header.seq % 30 == 0:
        print(f"L #{msg.header.seq:5d} | tags:{n:2d} | "
              f"err_now:{err if err else 0:.3f}px | "
              f"err_avg:{np.mean(err_left):.3f}px | "
              f"err_max:{np.max(err_left):.3f}px | "
              f"n_avg:{np.mean(count_left):.1f}")

    cv2.imshow("Left  (raw + detection)", vis)


def right_cb(msg):
    img = bridge.imgmsg_to_cv2(msg, "mono8")
    err, n, vis = detect_and_reproject(img, CAM1["K"], CAM1["D"], "R")
    if err is not None:
        err_right.append(err)
        count_right.append(n)

    if len(err_right) > 0 and msg.header.seq % 30 == 0:
        print(f"R #{msg.header.seq:5d} | tags:{n:2d} | "
              f"err_now:{err if err else 0:.3f}px | "
              f"err_avg:{np.mean(err_right):.3f}px | "
              f"err_max:{np.max(err_right):.3f}px | "
              f"n_avg:{np.mean(count_right):.1f}")

    cv2.imshow("Right (raw + detection)", vis)


def main():
    rospy.init_node("calib_validator", anonymous=True)
    print("=" * 65)
    print("  Stereo Calibration Validator")
    print(f"  Board: 6x6 AprilGrid, tag={TAG_SIZE*1000:.0f}mm, "
          f"spacing={TAG_SPACING*1000:.1f}mm")
    print(f"  Waiting for AprilGrid...  (Ctrl+C to stop)")
    print("=" * 65)

    rospy.Subscriber("/oak/left/image_raw", Image, left_cb, queue_size=2)
    rospy.Subscriber("/oak/right/image_raw", Image, right_cb, queue_size=2)

    rospy.spin()


if __name__ == "__main__":
    main()
