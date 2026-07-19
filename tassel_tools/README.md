# Tassel 可视化器

Tassel 可视化器通过 ROS 2 和 Foxglove 展示 VIO 的图像、轨迹、路标和优化状态。

## 功能

- 发布单目特征跟踪图、优化轨迹、IMU 里程计和路标点云。
- 发布优化代价、视觉因子数量等后端诊断信息。
- 自动启动 `foxglove_bridge`，安装 Tassel 可视化布局并打开 Foxglove Studio。

## 环境要求

- ROS 2 和 `foxglove_bridge`。
- Foxglove Studio。
- Python 3 和 `PyYAML`。

启动前加载 ROS 2 环境：

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
```

可视化器使用 `config/foxglove.yaml` 配置桥接端口、话题白名单和布局路径。

## 启动可视化器

在 Tassel 根目录执行：

```bash
python3 -m tassel_tools.viewer.foxglove config/foxglove.yaml
```

常用选项：

- `--no-bridge`：只安装布局并打开 Foxglove，不启动桥接。
- `--no-studio`：只启动桥接，不自动打开 Foxglove Studio。
- `--generate-only`：只生成并安装布局，不启动桥接和界面。
- `python3 -m tassel_tools.viewer.foxglove --help`：查看完整帮助。

默认桥接地址为：

```text
ws://127.0.0.1:8765
```

## 可视化配置

- `bridge.port`：Foxglove WebSocket 桥接端口，默认 `8765`。
- `bridge.address`：桥接服务监听地址，默认仅监听本机 `127.0.0.1`。
- `bridge.topic_whitelist`：允许转发到 Foxglove 的 ROS 2 话题列表。
- `studio.executable`：Foxglove Studio 可执行文件路径。
- `studio.layout_file`：Tassel Foxglove 布局文件路径。
- `studio.layout_name`：布局在 Foxglove 中显示的名称。
- `viewer.path_max_poses`：可视化器发布的轨迹最多保留的位姿数量。

## 发布话题

![Foxglove 可视化界面](../media/foxglove_overview_annotated.png)

图中箭头对应的可视化区域：

- `3D trajectory / landmarks`：`/vo/path`、`/ground_truth/path`、`/landmarks` 和 `/odom/camera`。
- `mono image / feature tracking`：`/mono/image`。
- `optimization plots`：`/optimization/*` 优化代价和视觉因子统计。

- `/mono/image`：左目单目特征跟踪图。
- `/odom/camera`：优化后的 IMU/机体里程计。
- `/vo/path`：视觉惯性优化轨迹。
- `/ground_truth/path`：数据集真值轨迹（如果存在）。
- `/landmarks`：当前窗口路标点云。
- `/optimization/total_reduction`：总优化代价变化。
- `/optimization/visual_reduction`：视觉代价变化。
- `/optimization/imu_reduction`：IMU 代价变化。
- `/optimization/prior_reduction`：边缘化先验代价变化。
- `/optimization/visual_factors_per_frame`：每帧视觉因子数量。
