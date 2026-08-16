# Tassel

Tassel 是一个研究型单目视觉惯性里程计，重点关注滑动窗口估计中的可观测性、先验管理、
边缘化一致性和实时后端优化。

## 演示视频

[![Tassel VIO 演示](media/bilibili_preview.jpg)](https://www.bilibili.com/video/BV1ukGA6sEuq/?vd_source=db129daae448e5d2371829d459869e86)

## 项目特色

- **单目视觉惯性估计**：包含 SFM、惯性对齐、预积分、重投影因子和滑动窗口优化。
- **时间延迟估计**：将相机-IMU 时间偏移纳入运动补偿和后端优化。
- **关键帧与选择性边缘化**：根据视觉连接关系维护路标宿主，并控制窗口中的状态规模。
- **FEJ 先验管理**：帧、特征和时间延迟分别保存 current 与首次线性化状态，边缘化先验
  使用 current residual 和冻结 Jacobian 构造固定坐标中的仿射模型。
- **Gauge 管理**：每轮以优化前保留槽的 current 位姿为参考，协同规范 current、FEJ 坐标
  表示和冻结先验的世界系列，保持 prior residual 与物理线性化身份不变。
- **Custom Ceres 后端**：复用视觉中间结果并特化固定结构 Schur 路径，同时保留 Ceres 的
  非线性优化、信赖域和步长接受机制。
- **数据集评估**：提供独立的 EuRoC 离线轨迹评估入口。

## 项目框架

```text
Tassel
├── tassel_core/                # 局部视觉惯性估计器
│   ├── cam/                    # 相机模型与投影接口
│   ├── estimator/              # 传播、优化、边缘化和窗口管理
│   ├── evaluation/             # EuRoC 等数据集评估入口
│   ├── factor/                 # 视觉、IMU、先验因子和视觉缓存
│   ├── frond_end/              # 特征跟踪、路标管理和三角化
│   ├── initial/                # SFM、惯性对齐和初始化
│   ├── marg/                   # 平方根边缘化和冻结先验
│   ├── solver/                 # 线性化系统和参数管理
│   ├── state/                  # 物理状态、优化参数缓存和 FEJ 线性化点
│   └── tests/                  # 单元测试
├── tassel_tools/               # 参数、ROS 2 和 Foxglove 工具
│   ├── parameters/             # 配置解析
│   ├── viewer/                 # 可视化和话题发布
│   └── tests/                  # 工具测试
├── tassel_utils/               # 通用数学和类型工具
├── third_party/ceres-solver/   # 面向 Tassel 的 Custom Ceres
├── cmake/                      # 依赖和测试配置
├── config/                     # 估计器和可视化配置
├── scripts/                    # 环境和数据集工具
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

评估程序读取单目相机、IMU 和真值文件，并对最终轨迹执行单一全局 yaw 与平移对齐，输出
ATE RMSE、终点位置误差和旋转 RMSE。第三个 `replay_hz` 参数控制实时回放频率。运行时
同时发布单目图像、特征追踪图像、IMU/相机里程计、轨迹、偏置和视觉因子窗口。可选的
第四个参数是 RTAB-Map 数据库路径；传入后会将保留关键帧提交给 RTAB-Map 后端。回放
结束时还会输出最后一次有效后验的 current/FEJ bias，不能读取窗口迁移后的 invalid 尾槽。

```bash
./build/tassel_core/test_euroc \
  config/euroc.yaml \
  datasets/machine_hall/MH_01_easy \
  60
```

完整调用形式为：

```text
test_euroc [config.yaml] [sequence_dir] [replay_hz] [rtabmap_database]
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
