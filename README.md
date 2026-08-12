# Tassel

Tassel 是一个研究型单目视觉惯性里程计，重点关注滑动窗口估计中的可观测性、先验管理、
边缘化一致性和实时后端优化。

## 演示视频

[![Tassel VIO 演示](media/bilibili_preview.jpg)](https://www.bilibili.com/video/BV1ukGA6sEuq/?vd_source=db129daae448e5d2371829d459869e86)

## 项目特色

- **单目视觉惯性估计**：包含 SFM、惯性对齐、预积分、重投影因子和滑动窗口优化。
- **时间延迟估计**：将相机-IMU 时间偏移纳入运动补偿和后端优化。
- **关键帧与选择性边缘化**：根据视觉连接关系维护路标宿主，并控制窗口中的状态规模。
- **保守状态管理**：低速时保护已有加速度计偏置先验，避免不可辨识方向错误吸收信息。
- **Schmidt/consider 先验**：保留固定变量均值、协方差及其与 active state 的交叉项。
- **Gauge 管理**：显式处理先验中的世界位置和 yaw 自由度，维护状态布局与协方差一致性。
- **Custom Ceres 后端**：复用视觉中间结果并特化固定结构 Schur 路径，同时保留 Ceres 的
  非线性优化、信赖域和步长接受机制。
- **可复现实验**：提供 EuRoC 轨迹评估、阶段计时、交错 A/B 性能测试和结果归档脚本。

## 项目框架

```text
Tassel
├── tassel_core/                # 局部视觉惯性估计器
│   ├── behavior/               # 估计行为和低速状态管理
│   ├── cam/                    # 相机模型与投影接口
│   ├── estimator/              # 传播、优化、边缘化和窗口管理
│   ├── factor/                 # 视觉、IMU、先验因子和视觉缓存
│   ├── frond_end/              # 特征跟踪、路标管理和三角化
│   ├── initial/                # SFM、惯性对齐和初始化
│   ├── marg/                   # 边缘化、gauge 和 Schmidt/consider 管理
│   ├── profiling/              # VTune 等性能分析标记
│   ├── solver/                 # 线性化系统和参数管理
│   ├── state/                  # 状态、变量角色和布局映射
│   └── tests/                  # 单元测试和 EuRoC 评估入口
├── tassel_tools/               # 参数、ROS 2 和 Foxglove 工具
│   ├── parameters/             # 配置解析
│   ├── viewer/                 # 可视化和话题发布
│   └── tests/                  # 工具测试
├── tassel_utils/               # 通用数学和类型工具
├── third_party/ceres-solver/   # 面向 Tassel 的 Custom Ceres
├── cmake/                      # 依赖和测试配置
├── config/                     # 估计器和可视化配置
├── scripts/                    # 环境、基准和性能分析脚本
├── media/                      # 演示素材
└── doc/                        # 理论和实验说明
```

## 数据集测试

### 依赖

- Ubuntu 22.04
- GCC 11
- CMake 3.22 或更新版本
- ROS 2 Humble
- Eigen 3.4、OpenCV、Sophus、Ceres 2.2、GTSAM、DBoW3

### 编译

```bash
source scripts/setup_environment.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j5
```

### EuRoC

测试程序读取单目相机、IMU 和真值文件，并对最终轨迹执行单一全局 yaw 与平移对齐，输出
ATE RMSE、终点位置误差和旋转 RMSE。第三个参数是离线回放频率，程序默认处理完整序列。

```bash
./build/tassel_core/test_euroc \
  config/euroc.yaml \
  datasets/machine_hall/MH_01_easy \
  60
```

## 参考文献

1. Li M, Mourikis A I. Online temporal calibration for camera-IMU systems: Theory and algorithms[J]. *The International Journal of Robotics Research*, 2014, 33(7): 947-964.
2. Qin T, Li P, Shen S. VINS-Mono: A robust and versatile monocular visual-inertial state estimator[J]. *IEEE Transactions on Robotics*, 2018, 34(4): 1004-1020.
3. Geneva P, Eckenhoff K, Lee W, et al. OpenVINS: A research platform for visual-inertial estimation[C]. *IROS*, 2020.
4. Usenko V, Demmel N, Schubert D, et al. Visual-inertial mapping with non-linear factor recovery[J]. *IEEE Robotics and Automation Letters*, 2020, 5(2): 422-429.
5. Dong-Si T C, Mourikis A I. Closed-form solutions for vision-aided inertial navigation[R]. University of California, Riverside, 2011.
6. Martinelli A. Observability properties and deterministic algorithms in visual-inertial structure from motion[R/OL]. 2014.
7. Campos C, Montiel J M M, Tardos J D. Inertial-only optimization for visual-inertial initialization[C]. *ICRA*, 2020.
8. Hailu H, Gebregziabher B. Motion as a sensing modality for metric scale in monocular visual-inertial odometry[EB/OL]. arXiv:2603.26740, 2026.
9. 高翔, 张涛, 刘毅, 等. 视觉 SLAM 十四讲：从理论到实践[M]. 第2版. 北京: 电子工业出版社, 2019.
10. 高翔. 自动驾驶与机器人中的 SLAM 技术：从理论到实践[M]. 北京: 电子工业出版社, 2023.
11. Zhang J, Kaess M, Singh S. On degeneracy of optimization-based state estimation problems[C]. *ICRA*, 2016.
12. Wu K J, Roumeliotis S I. Unobservable directions of VINS under special motions[R]. MARS Technical Report, 2016.
13. Gander W, Golub G H, von Matt U. A constrained eigenvalue problem[R]. ETH Zurich Technical Report, 1988.
14. Labbé M, Michaud F. RTAB-Map as an open-source lidar and visual simultaneous localization and mapping library[J]. *Journal of Field Robotics*, 2019, 36(2): 416-446.

## 版权

Copyright (c) 2026 Wu JunPing. 本项目采用 [MIT License](LICENSE)。参考项目和第三方代码的
版权、许可证及归属说明见 [NOTICE.md](NOTICE.md)。

作者：Wu JunPing · [GitHub](https://github.com/Ju-yzp) · `wjpzy20230551@qq.com`
