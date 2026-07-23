# Tassel Copyright and Attribution Notice

Copyright (c) 2026 Wu JunPing.

Author: Wu JunPing
Contact: wjpzy20230551@qq.com
Repository: https://github.com/Ju-yzp/Tassel

Tassel is released under the MIT License. It is an independent research
implementation that draws architectural and algorithmic inspiration from the
following open-source and academic SLAM/VIO systems:

- Open-VINS
- Basalt
- VINS-Mono
- RTAB-Map

RTAB-Map is referenced for its loop-closure architecture, including visual
place recognition, temporal hypothesis filtering, geometric verification and
pose-graph integration. Tassel implements its own estimator and loop-closure
components and does not claim authorship of RTAB-Map or its source code.

Tassel also links to or uses third-party libraries including Eigen, OpenCV,
Ceres Solver, Sophus, spdlog, yaml-cpp, DBoW3, GTSAM, ROS 2 and Fast-CDR.
These projects are governed by their own licenses and copyright notices.

Their respective licenses and copyright notices remain applicable to their own
source code and are not replaced by this notice.
