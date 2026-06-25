# Tassel

Tassel是学士帽上的流苏，这寓意着这是送给作者2027年毕业的礼物。

## 项目框架

```text
Tassel
├── tassel_core/                 # 核心估计模块
│   ├── frond_end/              # 特征跟踪、视差判定、三角化、异常点剔除
│   ├── factor/                 # 视觉因子、IMU因子、先验因子、预积分器
│   ├── initial/                # SFM、偏置估计、重力对齐、尺度恢复
│   ├── marg/                   # 路标块、IMU块、平方根边缘化、先验构建
│   ├── state/                  # 滑窗状态、参数块组织与转换
│   ├── cam/                    # 相机模型与投影接口
│   ├── estimator/              # 测量输入、状态传播、优化与滑窗管理
│   └── tests/                  # 核心模块测试
├── tassel_tools/               # 配套工具模块
│   ├── parameters/             # YAML 配置读取与参数组织
│   ├── viewer/                 # 图像、轨迹、点云等可视化功能
│   └── tests/                  # 工具模块测试
├── tassel_utils/               # 通用头文件工具库
│   ├── constants.h             # 常量定义
│   ├── types.h                 # 基础类型定义
│   ├── macros.h                # 断言与通用宏
│   ├── rotation.h              # 旋转相关工具
│   ├── se3_right_manifold.h    # SE(3) 流形参数化
│   ├── triangulation.h         # 三角化辅助函数
│   └── timer.h                 # 计时工具
├── config/                     # 项目运行配置
├── media/                      # 当前实现结果与演示素材
└── doc/                        # 论文与技术资料
```

## 阶段进展

| 日期 | 实现成果 | 展示 | 当前未解决问题 |
| --- | --- | --- | --- |
| 2026-06-23 | 基本完成vio,实现多传感器时间延迟在线估计等功能 | ![](media/VID_20260623232412.gif) | 消费级MEMS初始化、退化运动下的鲁棒性以及静止下的预积分器缓冲区数据累计。 |

## 参考文献

1. Li M, Mourikis A I. Online temporal calibration for camera–IMU systems: Theory and algorithms[J]. *The International Journal of Robotics Research*, 2014, 33(7): 947–964.

2. Qin T, Li P, Shen S. VINS-Mono: A robust and versatile monocular visual-inertial state estimator[J]. *IEEE Transactions on Robotics*, 2018, 34(4): 1004–1020.

3. Geneva P, Eckenhoff K, Lee W, et al. OpenVINS: A research platform for visual-inertial estimation[C]. *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2020.

4. Usenko V, Demmel N, Schubert D, et al. Visual-inertial mapping with non-linear factor recovery[J]. *IEEE Robotics and Automation Letters*, 2020, 5(2): 422–429.

5. Dong-Si T C, Mourikis A I. Closed-form solutions for vision-aided inertial navigation[R]. University of California, Riverside, 2011.

6. Martinelli A. Observabilty properties and deterministic algorithms in visual-inertial structure from motion[R/OL]. 2014.

7. Campos C, Montiel J M M, Tardos J D. Inertial-only optimization for visual-inertial initialization[C]. *2020 IEEE International Conference on Robotics and Automation (ICRA)*, 2020.

8. Hailu H, Gebregziabher B. Motion as a sensing modality for metric scale in monocular visual-inertial odometry[EB/OL]. arXiv:2603.26740, 2026.

9. 高翔, 张涛, 刘毅, 等. 视觉SLAM十四讲：从理论到实践[M]. 第2版. 北京: 电子工业出版社, 2019.

10. 高翔. 自动驾驶与机器人中的SLAM技术：从理论到实践[M]. 北京: 电子工业出版社, 2023.
