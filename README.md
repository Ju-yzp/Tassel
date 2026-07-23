# Tassel

Tassel 是学士帽上的流苏，这寓意着这是送给作者 2027 年毕业的礼物。

![Tassel VIO 演示](media/2026-07-18-03-02-23.gif)

[在 Bilibili 查看完整演示视频](https://www.bilibili.com/video/BV1TqKq6NE3t/?vd_source=db129daae448e5d2371829d459869e86)

## 核心特点

局部估计器主要参考 Open-VINS、VINS-Mono 和 Basalt 的相关设计；回环候选管理、
时序假设过滤、几何验证和位姿图组织参考了 RTAB-Map 的成熟机制。Tassel 保持独立实现，
参考项目的版权和许可证归其原作者所有。

- **在线时间延迟估计**：通过[时间延迟运动补偿模型](doc/time_delay_motion_model.md)，
  将相机-IMU 时间偏移纳入优化。
- **关键帧宿主生命周期管理**：延长以关键帧为宿主的路标生命周期，在窗口容量受限的情况下，保留局部轨迹约束。
- **选择性边缘化**：边缘化 IMU 因子及其关联的非关键帧视觉因子，避免无约束帧持续累积，降低静止状态下的速度漂移。
- **SFM 与惯性对齐初始化**：先通过[单目 SFM](doc/sfm_initialization.md)恢复无尺度
  视觉轨迹，再通过[惯性对齐](doc/inertial_alignment.md)顺序估计陀螺仪偏置、尺度、
  重力和速度。
- **异步回环组件**：估计器提交位姿、关键帧和历史路标事务，回环组件在独立任务中完成
  BRIEF/DBoW3 检索、时序后验门控、PnP 几何验证和 GTSAM 位姿图优化。
- **局部与全局解耦**：回环结果用于重建全局轨迹，不直接修改 VIO 滑窗状态、边缘化先验
  或既有线性化点。
- **对极几何因子预留**：源码中已包含对极几何因子，后续将继续完成与当前单目初始化流程的适配。

## 当前边界

- 当前回环使用 BRIEF 描述子和 DBoW3 词典，词典需要提前训练。
- 回环约束必须通过历史深度路标 PnP 验证；仅有图像相似度不会写入位姿图。
- 全局轨迹接受回环修正，局部 VIO 保持连续但不回写全局修正。
- 动态场景、强光照变化、弱纹理和长期地图管理仍需要继续验证。

## 项目框架

```text
Tassel
├── tassel_core/                # 局部视觉惯性估计器
│   ├── frond_end/              # 单目/多相机跟踪、路标管理、三角化与离群点剔除
│   ├── factor/                 # 重投影、IMU、先验因子及预积分器
│   ├── initial/                # SFM、偏置估计、重力对齐、尺度恢复
│   ├── marg/                   # 路标与状态平方根边缘化、先验构建
│   ├── state/                  # 帧物理状态、帧类型和优化参数转换
│   ├── cam/                    # 相机模型与投影接口
│   ├── estimator/              # 单目 VIO、状态传播、优化、边缘化与窗口管理
│   └── tests/                  # 核心模块测试
├── tassel_loop/                # 异步回环和全局位姿图
│   ├── loop_database.*         # BRIEF/DBoW3 检索与候选似然
│   ├── loop_hypothesis_tracker.* # 时序后验和候选门控
│   ├── pnp_verifier.*          # 历史路标 PnP 几何验证
│   ├── pose_graph.*            # GTSAM 里程计边与回环边优化
│   ├── trajectory_corrector.*  # 全局关键帧修正传播到稠密轨迹
│   └── tests/                  # 回环模块测试
├── tassel_tools/               # 配套工具模块
│   ├── parameters/             # YAML 配置读取与参数组织
│   ├── viewer/                 # ROS 2 话题发布与 Foxglove 启动器
│   └── tests/                  # 工具模块测试
├── tassel_utils/               # 通用头文件工具库
├── cmake/                      # 依赖查找、编译选项和测试辅助函数
├── config/                     # 估计器和可视化器配置
├── scripts/                    # 环境配置及辅助脚本
├── media/                      # 当前实现结果与演示素材
└── doc/                        # 理论推导、论文与技术资料
```

## 环境配置

当前验证环境：

- Ubuntu 22.04
- GCC 11
- CMake 3.22
- ROS 2 Humble
- Eigen 3.4
- OpenCV 4.11（通过 `TASSEL_VISION_PREFIX` 使用统一构建）
- Ceres Solver 2.2

项目依赖：

- Eigen3
- Ceres
- Sophus
- spdlog
- yaml-cpp
- OpenCV contrib（BRIEF）
- DBoW3
- GTSAM
- Fast CDR
- ROS 2
- TF2

OpenCV、DBoW3 和 TBB 必须来自兼容的构建；Eigen、Ceres 和 GTSAM 也必须使用一致的
Eigen ABI。相关 CMake 入口为 `TASSEL_VISION_PREFIX`、`TASSEL_MATH_PREFIX` 和
`TASSEL_EIGEN_DIR`。

仓库提供了环境配置脚本：

```bash
cd ~/Tassel

# 安装 Ubuntu/ROS 软件包，并加载当前终端环境
source scripts/setup_environment.sh --install

# 软件包已经安装时，只加载 ROS 和 CMake 环境
source scripts/setup_environment.sh
```

脚本不会自动编译 Sophus。Sophus 需要安装到 `/usr/local`、`~/.local`，或者通过
`SOPHUS_ROOT` 指定：

```bash
export SOPHUS_ROOT=$HOME/third_party/Sophus/install
source scripts/setup_environment.sh
```

## 编译

不使用硬件相机时：

```bash
source scripts/setup_environment.sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

## EuRoC 数据集运行

当前 EuRoC 测试使用单目相机，只读取 `mav0/cam0/data.csv` 和
`mav0/imu0/data.csv`。如果存在真值，还会读取
`mav0/state_groundtruth_estimate0/data.csv` 用于轨迹对比。

运行命令为：

```bash
./build/tassel_core/test_euroc \
  config/euroc.yaml \
  datasets/machine_hall/MH_01_easy \
  600 \
  20
```

参数依次为：

- `config/euroc.yaml`：EuRoC 相机、IMU 和估计器参数。
- `datasets/machine_hall/MH_01_easy`：序列目录。
- `600`：最大处理帧数，设置为 `0` 时运行完整序列。
- `20`：数据集回放频率，增大该值可以加速离线测试。

程序运行时会输出：

- SFM、陀螺仪偏置、重力方向和尺度初始化结果。
- 回环候选、PnP 验证、拒绝原因和位姿图优化结果（启用词典时）。
- 最终处理帧数、窗口最新帧索引和最终位姿。

例如，使用短序列快速验证数据流：

```bash
./build/tassel_core/test_euroc \
  config/euroc.yaml \
  datasets/machine_hall/MH_03_medium \
  300 \
  100
```

正常运行时会输出 `[pose]`、`[EuRoC] processed`和最终位姿；启用词典后还会输出
`[loop_verified]`、`[loop_rejected]`和最终位姿图统计。序列开头运动不足时，SFM 失败日志
表示初始化正在等待足够的视觉与惯性激励，并非数据读取失败。

## 回环运行

先使用与运行序列不同的数据训练 BRIEF 词典：

```bash
./build/tassel_loop/train_brief_vocabulary \
  "$HOME/.local/share/tassel/brief.dbow3" 400 \
  datasets/machine_hall/MH_02_easy \
  datasets/machine_hall/MH_03_medium \
  datasets/machine_hall/MH_04_difficult \
  datasets/machine_hall/MH_05_difficult
```

将词典作为第五个参数运行完整序列：

```bash
./build/tassel_core/test_euroc \
  config/euroc.yaml \
  datasets/machine_hall/MH_01_easy \
  0 \
  20 \
  "$HOME/.local/share/tassel/brief.dbow3"
```

回环流水线为：

```text
关键帧图像和历史深度路标
  -> BRIEF/DBoW3 候选检索
  -> 时序后验和似然比门控
  -> 多候选 PnP 几何验证
  -> GTSAM BetweenFactor<Pose3>
  -> 全局关键帧和稠密轨迹修正
```

该组织方式参考 RTAB-Map 的“外观候选、时序假设、几何验证、图优化”分层机制，
Tassel 使用自己的关键帧事务、历史路标和局部/全局轨迹接口实现。更详细的话题与 Foxglove
说明见 [tassel_tools/README.md](tassel_tools/README.md)。

## 可视化器

可视化器使用 ROS 2 `foxglove_bridge` 和 Foxglove Studio，具体环境、配置、启动参数和
话题说明见 [tassel_tools/README.md](tassel_tools/README.md)。

启动可视化器：

```bash
cd ~/Tassel
source scripts/setup_environment.sh
python3 -m tassel_tools.viewer.foxglove config/foxglove.yaml
```

启动器会自动安装项目布局、启动 `foxglove_bridge` 并打开 Foxglove Studio。修改
`config/foxglove.yaml` 后需要重启启动器，已经运行的 Bridge 不会自动更新白名单。

## 理论文档

- [相机-IMU 时间延迟运动补偿模型](doc/time_delay_motion_model.md)
- [单目 SFM 初始化机制](doc/sfm_initialization.md)
- [单目 SFM 与惯性对齐](doc/inertial_alignment.md)

文档公式与当前代码变量、坐标系和符号约定对应。惯性对齐文档同时记录了仍需通过合成
数据验证的外参常数项，不将待核对实现作为已成立结论。

## 作者与引用

- 作者：Wu JunPing
- 邮箱：`wjpzy20230551@qq.com`
- GitHub：[Ju-yzp](https://github.com/Ju-yzp)
- 仓库：[github.com/Ju-yzp/Tassel](https://github.com/Ju-yzp/Tassel)

学术工作使用本项目时，请参考 [CITATION.cff](CITATION.cff)。版权、参考项目和第三方
归属说明见 [NOTICE.md](NOTICE.md)。问题报告应包含数据集、配置、运行命令、提交版本和
可复现日志。

## 参考文献

1. Li M, Mourikis A I. Online temporal calibration for camera-IMU systems: Theory and algorithms[J]. *The International Journal of Robotics Research*, 2014, 33(7): 947-964.

2. Qin T, Li P, Shen S. VINS-Mono: A robust and versatile monocular visual-inertial state estimator[J]. *IEEE Transactions on Robotics*, 2018, 34(4): 1004-1020.

3. Geneva P, Eckenhoff K, Lee W, et al. OpenVINS: A research platform for visual-inertial estimation[C]. *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2020.

4. Usenko V, Demmel N, Schubert D, et al. Visual-inertial mapping with non-linear factor recovery[J]. *IEEE Robotics and Automation Letters*, 2020, 5(2): 422-429.

5. Dong-Si T C, Mourikis A I. Closed-form solutions for vision-aided inertial navigation[R]. University of California, Riverside, 2011.

6. Martinelli A. Observabilty properties and deterministic algorithms in visual-inertial structure from motion[R/OL]. 2014.

7. Campos C, Montiel J M M, Tardos J D. Inertial-only optimization for visual-inertial initialization[C]. *2020 IEEE International Conference on Robotics and Automation (ICRA)*, 2020.

8. Hailu H, Gebregziabher B. Motion as a sensing modality for metric scale in monocular visual-inertial odometry[EB/OL]. arXiv:2603.26740, 2026.

9. 高翔, 张涛, 刘毅, 等. 视觉 SLAM 十四讲：从理论到实践[M]. 第2版. 北京: 电子工业出版社, 2019.

10. 高翔. 自动驾驶与机器人中的 SLAM 技术：从理论到实践[M]. 北京: 电子工业出版社, 2023.

11. Zhang J, Kaess M, Singh S. On degeneracy of optimization-based state estimation problems[C]. *2016 IEEE International Conference on Robotics and Automation (ICRA)*, 2016.

12. Wu K J, Roumeliotis S I. Unobservable directions of VINS under special motions[R]. University of Minnesota, MARS Laboratory Technical Report 2016-002, 2016.

13. Gander W, Golub G H, von Matt U. A constrained eigenvalue problem[R]. ETH Zurich, D-INFK Technical Report 92, 1988.

14. Labbé M, Michaud F. RTAB-Map as an open-source lidar and visual simultaneous localization and mapping library for large-scale and long-term online operation[J]. *Journal of Field Robotics*, 2019, 36(2): 416-446.

## 许可证

Copyright (c) 2026 Wu JunPing. 项目使用 [MIT License](LICENSE)。版权、作者、参考项目
和第三方归属说明见 [NOTICE.md](NOTICE.md)。
