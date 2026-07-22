# Tassel

Tassel 是学士帽上的流苏，这寓意着这是送给作者 2027 年毕业的礼物。

![Tassel VIO 演示](media/2026-07-18-03-02-23.gif)

[在 Bilibili 查看完整演示视频](https://www.bilibili.com/video/BV1TqKq6NE3t/?vd_source=db129daae448e5d2371829d459869e86)

## 核心特点

Tassel 是一个面向单目视觉惯性里程计的研究型实现，主要吸收了 Open-VINS、VINS-Mono
和 Basalt 中的相关设计思想。

- **在线时间延迟估计**：通过时间延迟运动补偿模型，将相机-IMU 时间偏移纳入优化。
- **关键帧宿主生命周期管理**：延长以关键帧为宿主的路标生命周期，在窗口容量受限的情况下，保留局部轨迹约束。
- **选择性边缘化**：边缘化 IMU 因子及其关联的非关键帧视觉因子，避免无约束帧持续累积，降低静止状态下的速度漂移。
- **联合 SFM 初始化**：在对极几何约束中联合优化并估计陀螺仪初始偏置，提升单目初始化精度；平移方向可观，但位移尺度对该过程不敏感。
- **对极几何因子预留**：源码中已包含对极几何因子，后续将继续完成与当前单目初始化流程的适配。

## 发展方向

后续将引入回环检测与全局修正，进一步提升系统在高动态、低纹理和高频振动等极端环境下的鲁棒性。

项目地址：[github.com/Ju-yzp/Tassel](https://github.com/Ju-yzp/Tassel)

## 项目框架

```text
Tassel
├── tassel_core/                # 核心估计模块
│   ├── frond_end/              # 单目/多相机跟踪、路标管理、三角化与离群点剔除
│   ├── factor/                 # 重投影、IMU、先验因子及预积分器
│   ├── initial/                # SFM、偏置估计、重力对齐、尺度恢复
│   ├── marg/                   # 路标消元、平方根边缘化与先验构建
│   ├── state/                  # 帧状态、保留宿主槽位及参数转换
│   ├── cam/                    # 相机模型与投影接口
│   ├── estimator/              # 单目 VIO、状态传播、优化、边缘化与窗口管理
│   └── tests/                  # 核心模块测试
├── tassel_tools/               # 配套工具模块
│   ├── parameters/             # YAML 配置读取与参数组织
│   ├── viewer/                 # ROS 2 话题发布与 Foxglove 启动器
│   └── tests/                  # 工具模块测试
├── tassel_utils/               # 通用头文件工具库
├── config/                     # 估计器和可视化器配置
├── scripts/                    # 环境配置及辅助脚本
├── media/                      # 当前实现结果与演示素材
└── doc/                        # 论文与技术资料
```

## 环境配置

当前验证环境：

- Ubuntu 22.04
- GCC 11
- CMake 3.22
- ROS 2 Humble
- 系统 OpenCV 4.5.4

项目依赖：

- Eigen3
- Ceres
- Sophus
- spdlog
- yaml-cpp
- Fast CDR
- ROS 2
- TF2

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

- 每帧特征输入、连续跟踪、新增特征、关键帧连接率和视差。
- SFM、陀螺仪偏置、重力方向和尺度初始化结果。
- 视觉、IMU、边缘化先验在优化前后的代价。
- 加速度计偏置、陀螺仪偏置和相机-IMU 时间延迟估计。
- 重投影离群点数量、保留宿主帧和边缘化先验尺寸。
- 最终处理帧数、窗口最新槽位和最终位姿。

例如，使用短序列快速验证数据流：

```bash
./build/tassel_core/test_euroc \
  config/euroc.yaml \
  datasets/machine_hall/MH_03_medium \
  300 \
  100
```

正常运行时应持续出现 `mono frames`、优化代价和离群点统计；序列开头运动不足时，
`SFM: no suitable parallax frame` 表示初始化正在等待足够视差，并非数据读取失败。

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

## 许可证

项目使用 [MIT License](LICENSE)。版权、作者和参考项目说明见 [NOTICE.md](NOTICE.md)。
