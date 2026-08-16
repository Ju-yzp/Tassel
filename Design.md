# Tassel VIO 后端重构设计

## 1. 文档目的

本文定义 Tassel 滑窗 VIO 后端的目标架构、数学契约、当前实现差距和分阶段迁移方案。

本次重构的核心不是管理层性能，而是建立以下长期契约：

- 状态、特征、预积分和 prior 的所有权不清晰；
- current、posterior 和 linearized 数据存在混用风险；
- 预积分端点依赖窗口下标；
- current residual、冻结 Jacobian 和 prior 仿射坐标必须严格一致；
- gauge 规范必须保持物理状态、FEJ 身份和 prior residual 等价；
- 保留槽依赖特殊数组下标；
- 特征换宿主后无法继续沿用旧逆深度 FEJ 身份；
- 零速检测通过启发式恢复 bias，缺少独立测量模型。

本文只定义目标和迁移顺序。生产代码必须按阶段修改，每阶段单独验证。

## 2. 已确定的设计决策

### 2.1 FEJ 范围

Tassel 采用以下策略：

- 活动窗口优化始终使用最新 current 状态和 current 特征深度；
- 状态首次进入优化问题时捕获 linearized 状态；
- 特征首次获得有效深度并进入优化问题时捕获 linearized 深度；
- 边缘化 residual 使用最新 current，状态和深度 Jacobian 使用冻结 linearized 数据；
- 生成 prior 后，参考状态、平方根 Jacobian、残差和列布局全部冻结；
- prior 不执行以 current 为新原点的 recenter；仅允许保持同一物理 FEJ 点的 gauge 坐标
  等价变换，并同步变换世界系 position/velocity 列；
- 每轮优化前保存 retained current 位姿，优化后将 current 规范回该参考。

这不是“所有活动因子永久冻结 Jacobian”。活动因子仍在最新 current 点重线性化，只有边缘化系统和历史 prior 使用冻结参考。

### 2.2 两份数值状态

每个帧和特征只维护两份核心数值：

```text
current
linearized
```

posterior 不是第三份数值缓存。它表示 current 已经是最近一次优化接受结果：

```text
has_posterior == true
```

新帧预测和三角化读取 posterior current。

### 2.3 放弃零速检测

当前重构不包含零速检测，不根据窗口速度恢复、固定或修改加速度 bias。

需要删除：

- `Estimator::isStationaryWindow()`；
- `kStationarySpeed`；
- `hold_accel_bias` 参数；
- `WindowOptimizer` 中保存和恢复加速度 bias 均值的逻辑。

将来若重新引入静止约束，必须作为具有明确噪声模型和触发条件的独立因子实现，不能在求解后覆盖优化结果。

### 2.4 特征换宿主

特征换宿主会改变逆深度参数化：

```text
rho_old: relative to old host camera
rho_new: relative to new host camera
```

因此不能继承旧 `linearized_inverse_depth`。

最终方案是：

1. 使用 current 位姿和 current depth 将几何转移到新宿主；
2. 继承仍有效的观测；
3. 创建新的特征 FEJ 身份；
4. 写入新的 current depth；
5. 清空 `has_linearized_depth`；
6. 新特征首次进入优化时重新捕获 linearized depth；
7. 旧特征身份退休。

特征身份必须包含稳定 `feature_id` 和 FEJ generation，不能只依赖容器索引。

### 2.5 多态帧容器

帧采用继承体系：

```text
Frame
├── NormalFrame
├── KeyFrame
└── RetainedFrame
```

活动窗口使用：

```cpp
std::vector<std::unique_ptr<Frame>> frames;
```

保留槽独立使用：

```cpp
std::unique_ptr<RetainedFrame> retained_frame;
```

不能使用 `std::vector<Frame>` 保存派生类型，否则会发生对象切片，虚函数行为和派生状态都会丢失。

## 3. 理论契约

### 3.1 状态和坐标约定

当前位姿参数布局保持：

```text
pose = [position(3), rotation_log(3)]
speed_bias = [velocity(3), accel_bias(3), gyro_bias(3)]
```

位姿使用 `SE3RightManifold`。所有边缘化增量必须使用与该流形一致的右扰动定义。

对冻结参考状态 `x0` 和 current 状态 `x`：

```text
delta = x minus x0
```

prior residual 为：

```text
r_prior = H * delta + b
```

`H` 是生成 prior 时的局部 Jacobian，不能被后续 current 状态改写。

### 3.2 Gauge 自由度

相机和 IMU 系统的目标 gauge 为：

- 全局平移 3 DOF；
- 绕重力方向的 yaw 1 DOF。

roll 和 pitch 由重力方向约束，不能固定完整参考位姿。

对全部 current 状态应用同一变换：

```text
p_i' = R_g * p_i + t_g
R_i' = R_g * R_i
v_i' = R_g * v_i
```

其中 `R_g` 只绕重力方向旋转。accel bias、gyro bias 和逆深度不随该 gauge 变换。

### 3.3 Gauge 参考

每轮 gauge 参考是优化开始前保留槽的 current 位姿快照：

```text
reference position = retained_frame.current_position_before_solve
reference yaw      = yaw(retained_frame.current_rotation_before_solve)
```

设保留帧优化后 current 位姿为 `(R_c, p_c)`，优化前快照为 `(R_r, p_r)`：

```text
yaw_delta = yaw(R_r) - yaw(R_c)
R_g       = rotation_about_gravity(yaw_delta)
t_g       = p_r - R_g * p_c
```

参考必须取自本轮求解前的 current，而不能切换到 retained frozen pose。规范化是内部世界
坐标变换，不是重新线性化。设 `Q=R_g`，则 current 与 linearized 的坐标表示都必须满足：

```text
p' = Q * (p - p_c) + p_r
R' = Q * R
v' = Q * v
```

右旋转扰动在全局左乘 `Q` 下保持不变；position 和 velocity 使用世界系加法增量：

```text
delta_p' = Q * delta_p
delta_v' = Q * delta_v
delta_theta' = delta_theta
```

为保持 `H * delta + b` 严格不变，prior 的 position/velocity 列必须右乘 `Q^T`，旋转、
bias、delay 列以及 `b` 不变。FEJ 物理点没有改变，改变的只是它在新世界系中的表示。

工程约束是：

- 每轮优化提交后立即将 current 规范回求解前 retained current gauge；
- 不允许把 retained linearized pose 当作新的 gauge 参考，否则 retained 身份切换会造成
  输出坐标跳变；
- current、linearized 和 prior 必须在一次事务中完成坐标变换；
- 有限 gauge 变换前后的 prior residual 必须在数值精度内相等；
- retained 长期对象与 `frames[0]` 优化镜像必须同步。

### 3.4 Gauge 奇异性

若 yaw 使用机体 x 轴在水平面的投影定义：

```text
heading_norm = hypot(R(0, 0), R(1, 0))
```

当 `heading_norm` 小于阈值时，机体 x 轴接近竖直，yaw 表达奇异，必须抛出异常。

还必须拒绝：

- 非有限的位置、旋转或速度；
- 不满足正交性或行列式不接近 1 的旋转；
- current 与 linearized 的 `frame_id` 不一致；
- 未建立 linearized 状态的保留帧；
- 非有限的 gauge 修正或变换结果。

状态提交必须先完整验证，再统一写入，不能留下部分帧已变换的状态。

### 3.5 FEJ 边缘化模型

经典 FEJ 冻结的是 Jacobian 的评估点，不是把 residual 永久停留在状态首次进入窗口时。若 residual 也始终在旧 FEJ 点计算，后续非线性优化得到的最新后验信息不会正确进入新 prior。

设冻结状态为 `x_f`，最新 current 为 `x_c`：

```text
delta_c = x_c minus x_f
r_c     = residual evaluated at current
J_f     = Jacobian evaluated at frozen FEJ state/depth
```

在固定 FEJ 坐标 `delta = x minus x_f` 中，新因子的一阶模型必须写成：

```text
r(delta) = J_f * delta + b_f
b_f      = r_c - J_f * delta_c
```

它同时满足：

```text
r(delta_c) = r_c
dr / d(delta) = J_f
```

视觉因子的 `J_f` 必须统一使用 linearized host pose、target pose、delay 和 inverse depth；IMU Jacobian同样使用两端 linearized 状态。`r_c` 使用对应的最新 current 状态和 current depth。

因此 current residual 与 linearized Jacobian 的组合不是无意混用，而是 FEJ 的明确数学定义。禁止的是：

- 没有计算 `-J_f * delta_c` 就直接把 `r_c` 当作固定 FEJ 坐标中的常数项；
- 一个 Jacobian 内混用 linearized pose 和 current depth；
- 使用 current visual cache 计算 `J_f`；
- 使用不同 FEJ generation 的状态和特征；
- residual 和 `delta_c` 读取不同版本的 current。

### 3.6 冻结 prior

一次边缘化结果由以下不可拆分数据组成：

```text
H
b
retained linearization states
linearized delay
variable layout
frame_id mapping
```

生成后物理 FEJ 身份和仿射模型全部冻结。禁止：

- recenter prior 到 current；
- 将某个 current 状态冒充新的 FEJ 点；
- 只变换 current 而不变换 FEJ 表示和 prior 世界系列；
- 单独替换参考点、`H` 或 `b`；
- 依赖已经删除的历史变量重新线性化。

Ceres 所需的 local-to-ambient Jacobian 转换只能写入本次 `Evaluate()` 输出，不能写回 `H`。
唯一允许修改 `H` 的操作是完整 gauge 坐标变换中的世界系列基变换；该操作必须与全部
linearization state 同步，并由 residual 不变性测试覆盖。

## 4. 对象模型

### 4.1 Frame 基类

`Frame` 拥有所有帧共有的数据：

```text
frame_id
timestamp
current pose / speed / ba / bg
linearized pose / speed / ba / bg
IMU sample at image time
image synchronization delay
current visual cache
lifecycle flags
```

基础标志至少包括：

```cpp
bool has_current_parameters_;
bool has_posterior_;
bool has_linearized_;
bool current_cache_valid_;
```

新代码统一使用 `linearized` 拼写。若迁移期必须兼容 `has_lineared_`，只能提供短期别名，不继续扩散该拼写。

基础接口直接返回数据指针，不引入参数聚合结构：

```cpp
virtual double* get_current_pose() = 0;
virtual double* get_current_speed() = 0;
virtual double* get_current_ba() = 0;
virtual double* get_current_bg() = 0;

virtual const double* get_linearized_pose() const = 0;
virtual const double* get_linearized_speed() const = 0;
virtual const double* get_linearized_ba() const = 0;
virtual const double* get_linearized_bg() const = 0;
```

基类必须有虚析构函数。

### 4.2 帧类型职责

`NormalFrame`：

- 普通活动状态；
- 可以被边缘化或释放；
- 不承担长期特征宿主语义。

`KeyFrame`：

- 可以作为长期特征宿主；
- 参与保留帧选择；
- 仍然是可优化状态，不是固定 gauge。

`RetainedFrame`：

- 独立于活动帧容器；
- 对应 prior 中的 retained state；
- 提供固定 linearized gauge；
- current 状态仍参与优化；
- 替换时必须创建新的对象身份，不能覆盖旧对象后沿用旧 FEJ。

关键帧类型必须在 Ceres `Problem` 建立前确定。不能在 `Problem` 存活期间替换对象导致参数地址失效。

### 4.3 Feature

特征拥有：

```text
feature_id
fej_generation
host_frame_id
observations keyed by frame_id
current_inverse_depth
linearized_inverse_depth
has_current_depth
has_linearized_depth
```

接口：

```cpp
double* get_current_inverse_depth();
const double* get_linearized_inverse_depth() const;
```

三角化和优化只更新 current depth。linearized depth 捕获后保持冻结。

换宿主创建新的 FEJ generation，旧 generation 不再参与后续优化或边缘化。

### 4.4 PreintegrationEdge

预积分显式描述物理端点：

```text
from_frame_id
to_frame_id
start_timestamp
end_timestamp
measurements
delta rotation / velocity / position
covariance
bias linearization point
```

不能通过 `preintegrators[i]` 推断其连接 `frames[i]` 和 `frames[i+1]`。

接口必须验证：

- 两端 `frame_id` 与请求状态一致；
- 时间严格递增；
- 测量区间与两端时间一致；
- bias 线性化点有限；
- 被窗口移除的端点不会继续被引用。

## 5. State 与数据流

### 5.1 State 所有权

`State` 拥有：

```cpp
size_t max_frame_num;
size_t current_frame_count;
std::vector<std::unique_ptr<Frame>> frames;
std::unique_ptr<RetainedFrame> retained_frame;
std::vector<std::unique_ptr<PreintegrationEdge>> imu_edges;
```

`current_frame_count` 表示活动容器中的有效帧数量，不包含保留帧：

```text
0 <= current_frame_count <= max_frame_num
latest active index = current_frame_count - 1
```

窗口满时禁止在 `Problem` 存活期间替换对象。完成优化、边缘化和窗口推进后，尾槽必须从
上一帧 posterior current 建立下一帧预测种子，并清除旧 `frame_id` 和 FEJ 身份：

```text
seed current pose / velocity / ba / bg from previous posterior
reset has_linearized
set frame_id invalid
capture a new FEJ point when the new frame first enters optimization
```

容器槽地址可以复用，Frame/FEJ 身份不能复用。只把尾槽 `frame_id` 设为 invalid 而保留
`has_linearized=true`，会使所有后续新帧永久使用该槽第一次占用者的线性化点。

### 5.2 原始数据入口

`Estimator` 只提交：

```text
frame_id / image timestamp
feature observations
IMU measurements
synchronization delay
```

`State` 根据当前帧数量：

1. 创建目标帧；
2. 选择上一活动帧；
3. 创建显式 `previous.frame_id -> current.frame_id` 预积分边；
4. 调用目标帧预测；
5. 验证成功后增加 `current_frame_count`。

`Estimator` 不直接写 pose、velocity、bias，不移动预积分数组，也不维护 FEJ 标志。

### 5.3 后验预测

预测从上一帧 posterior current 出发：

```text
previous posterior current
    + IMU interval
    -> new frame current
```

帧可以提供：

```cpp
virtual void predict_from(
    const Frame& previous,
    const PreintegrationEdge& edge) = 0;
```

调用前要求 `previous.has_posterior() == true`。预测不能读取 previous linearized 状态。

### 5.4 状态同步和提交

状态生命周期动作必须分开：

```cpp
void state_to_current_parameters();
void capture_linearized_state();
void current_parameters_to_state();
void normalize_current_gauge();
void mark_current_as_posterior();
void invalidate_current_cache();
```

对外可以提供一个事务式 `commit_optimized_state()`，但其内部顺序必须是：

1. 将优化参数解析到临时 current 状态；
2. 验证所有状态；
3. 以求解前保存的 retained current 位姿计算 gauge；
4. 规范全部 current pose 和 velocity；
5. 对全部 Frame FEJ 表示施加同一坐标变换；
6. 对 prior 的 FEJ 表示和世界系 position/velocity 列施加等价变换；
7. 同步 retained 长期对象和优化镜像；
8. 提交 current 并标记 posterior；
9. 失效 current cache。

不建议把 gauge 规范隐藏成单个帧 `paramToState()` 的副作用，因为 gauge 必须同时作用于整个窗口。它应由 `State` 的事务式提交统一完成。

## 6. 阶段数据选择

| 阶段 | 状态输入 | 深度输入 | 输出 |
| --- | --- | --- | --- |
| 新帧预测 | 上一帧 posterior current | 无 | 新帧 current |
| 新特征三角化 | 相关帧 posterior current | 无 | current depth |
| 活动窗口优化 | 最新 current | 最新 current depth | accepted current |
| 优化提交 | accepted current | accepted current depth | posterior 标志 |
| 边缘化 residual | 最新 current | 最新 current depth | `r_c` |
| 边缘化 Jacobian | linearized | linearized depth | `J_f` |
| FEJ 常数项 | current + linearized | current + linearized depth | `b_f = r_c - J_f * delta_c` |
| gauge 规范 | accepted current + retained current 快照 + FEJ/prior | 深度不变 | 等价新世界系表示 |

活动优化始终在最新点计算 residual 和 Jacobian。优化结果只写回 current。

边缘化始终读取 linearized getter，不允许上层自行复制或选择字段。

## 7. 帧缓存

帧缓存只描述 current 状态的派生量：

```text
current pose / speed / bias / delay
    -> delay-compensated pose
    -> rotation parameter Jacobian
    -> delay Jacobian
    -> current visual cache
```

current cache 必须绑定：

```text
frame_id
current evaluation version
```

以下事件使缓存失效：

- Ceres 新 current 评估点；
- current 参数被修改；
- 试探点接受或拒绝；
- 优化结果提交；
- gauge 规范；
- 窗口移动；
- 对象替换或 `frame_id` 改变。

边缘化 residual 可以读取同一 current 版本的 current cache；边缘化 Jacobian 禁止读取 current cache。Jacobian 必须使用单次边缘化私有的只读 linearized cache，并在 prior 生成后释放。

## 8. 优化器与边缘化器

### 8.1 WindowOptimizer

`WindowOptimizer`：

- 只获取 current 参数指针；
- 使用 current visual cache；
- 按显式预积分边取得两个端点；
- 将 accepted pose、speed、bias 和 depth 写回 current；
- 不捕获或修改 linearized 数据；
- 不执行零速检测；
- 不在求解后恢复 bias。

### 8.2 Marginalizer

`Marginalizer`：

- 同时获取版本一致的 current 和 linearized 参数指针；
- 在 current 点计算 `r_c`；
- 在 linearized 点计算 `J_f`；
- 计算 `delta_c` 和 `b_f = r_c - J_f * delta_c`；
- 使用显式变量布局映射 `frame_id -> column`；
- 使用显式预积分端点；
- residual 使用 current depth，Jacobian 使用 linearized depth；
- 独立计算 linearized 视觉中间量；
- 消元深度和退休状态；
- 生成新的冻结 `MargLinData`。

`Marginalizer` 可以读取已验证版本的 current cache 计算 `r_c`，但不能用它计算 `J_f`。linearized Jacobian 使用独立只读缓存。它不调用 `stateToParams()`，不修改 Frame 或 Feature。

### 8.3 MargLinData

`MargLinData` 保存：

```text
H
b
linearized state values
linearized delay
frame_id -> block mapping
variable layout
```

布局不能再依赖：

```text
frame0 is retained
frame1 is oldest
preintegrator i connects frame i and i+1
```

所有映射都必须显式建立并验证完整性。

## 9. 当前实现状态与剩余边界

### 9.1 prior 不重新定心，gauge 坐标协同变换

证据：

- `Estimator::optimize()` 直接读取冻结的 prior；
- `normalizeCurrentGauge()` 同步变换 current、Frame FEJ、retained 镜像和 prior；
- `MargLinData::transformGauge()` 变换 FEJ 表示并对 position/velocity 列执行基变换；
- gauge 不变性测试验证变换前后 `H * delta + b` 一致。

当前约束：

- prior 不 recenter 到 current，`b` 不因优化结果重写；
- gauge 变换不创建新 FEJ 身份，只改变同一物理点和 Jacobian 的坐标表示。

### 9.2 边缘化使用 current 数据

证据：

- `updateMarginalizationPrior()` 调用 `stateToParams()`；
- `LandmarkBlock` 读取 `param_pose`、`param_speed_bias` 和 `estimated_depth`；
- `MarginalizationSqrt` 使用 current `VisualFrameCache`；
- IMUBlock 从 current `FrameState` 字段线性化。

当前实现：

- Frame、Feature 和 delay 均保存 current/linearized 数据；
- 视觉和 IMU 边缘化块在 current 点计算 residual，在 FEJ 点计算 Jacobian；
- 固定坐标常数项统一使用 `b_f = r_c - J_f * delta_c`；
- 活动优化继续在 current 点正常重线性化。

### 9.3 旧 prior 使用固定 FEJ 仿射截距

reduced system 的列变量是相对 FEJ 点的全局增量，因此旧 prior 必须直接写入仿射截距
`b`，而不是写入 current residual。其 current residual 为：

```text
r_current = H * (current minus linearized) + b
```

写入边缘化系统前，prior 与 State 的 pose、velocity、ba、bg、delay 线性化点逐项验证。
任何身份不一致都抛出异常，不能静默组装，也不能只替换 `b`。

### 9.4 保留槽和布局依赖特殊下标

证据：

- `kRetainedFrameIndex = 0`；
- 多处直接访问 `frames[0]`；
- prior 列布局依赖 retained 位于首块。

问题：

- 状态身份、对象所有权和矩阵布局耦合；
- 替换保留帧时容易产生错位。

修改：

- retained 独立所有；
- 使用 `frame_id -> VariableBlock` 显式布局；
- 优化器和边缘化器按布局获取参数。

### 9.5 预积分端点由下标推断

证据：

- 优化器将 `preintegrators[i]` 连接 `frames[i]` 和 `frames[i+1]`；
- 滑窗通过移动 vector 元素迁移预积分；
- 边缘化通过 `first_imu_index` 推断端点。

问题：

- 槽位移动后端点语义隐含；
- 保留槽独立后无法继续依赖连续下标。

修改：

- 引入显式 `PreintegrationEdge(from_frame_id, to_frame_id)`；
- State 按 ID 选择端点；
- 滑窗删除边，不移动其物理含义。

### 9.6 预测由 Estimator 直接修改状态

证据：

- `Estimator::predictFrameState()` 直接读写帧 pose、position、velocity 和 bias；
- 通过下标选择预积分器。

问题：

- Estimator 承担状态内部行为；
- 预测输入没有由类型保证为 posterior current。

修改：

- 将预测行为移动到 Frame/State；
- Estimator 只提交原始 IMU；
- Frame 验证 previous posterior 和 edge 端点。

### 9.7 Feature 只有一个深度

证据：

- `Feature::estimated_depth` 同时用于三角化、优化和边缘化；
- `transferHost()` 直接覆盖该值。

问题：

- current 和 FEJ 深度无法区分；
- 换宿主后旧 linearized depth 不再属于同一参数化。

修改：

- 增加 current/linearized depth；
- 换宿主创建新的 FEJ generation；
- 旧身份退休。

### 9.8 Frame 继承与值容器冲突

如果采用 `NormalFrame/KeyFrame/RetainedFrame` 继承，却使用 `std::vector<Frame>`，会发生对象切片。

修改：

- 活动窗口使用 `std::vector<std::unique_ptr<Frame>>`；
- retained 使用独立 `std::unique_ptr<RetainedFrame>`；
- Ceres Problem 存活期间不替换对象。

### 9.9 独立 GaugeAnchor 已删除

当前实现不再保存 `State::gauge_reference`。每轮参考来自求解前 retained current 位姿快照；
retained 尚未建立时，使用首个活动帧的求解前 current 位姿。FEJ 点不承担输出坐标锚的职责。

运行时约束：

- retained frame ID 必须与 Ceres 窗口中的参考镜像一致；
- gauge 规范同步变换 current、linearized 表示和 prior 世界系列；
- bias、深度、delay 和 prior 常数项 `b` 不随 gauge 改变；
- retained 长期对象必须在规范后从 `frames[0]` 镜像同步。

### 9.10 尾槽复用必须创建新 FEJ 身份

活动窗口使用固定容量容器，但 FEJ 身份属于帧，不属于数组下标。窗口迁移完成后，尾槽通过
`seedFrameState(previous, tail)` 继承上一后验作为预测初值，同时清除 `has_linearized`。

该契约同时适用于初始化窗口滑动和正常边缘化窗口迁移。历史错误只清除了 `frame_id`，使
尾槽长期保留初始化阶段的零 `ba` FEJ 点，导致加速度 bias 无法收敛。修复后 MH_01 的结果为：

```text
estimated ba = (-0.02682, 0.13690, 0.08293)
ground truth ≈ (-0.02550, 0.13627, 0.07640)
ATE RMSE     = 0.333 m
rotation RMSE= 0.0269 rad
```

### 9.11 零速检测不构成严格测量模型

当前实现已经删除按窗口速度判断静止并在求解后恢复 accel bias 的启发式机制。

问题：

- 没有对应 residual、Jacobian、协方差和可观性分析；
- 求解器报告的 accepted 状态随后被人工修改；
- 优化模型和最终输出状态不一致。

当前约束：

- 不允许在求解器接受后覆盖 bias；
- 不提供替代启发式。

### 9.11 time delay 缺少双状态

当前只有 `time_delay/param_time_delay` 和 prior 内的 linearization delay。

问题：

- 活动优化和边缘化数据所有权不对称；
- 边缘化无法只通过统一 getter 获取 linearized delay。

修改：

- State 管理 current delay、linearized delay 和 `has_linearized_delay`；
- 优化器读取 current；
- Marginalizer 读取 linearized。

当前实现约定：

- `param_time_delay` 是 Ceres 的唯一时间延迟参数块；`getCurrentTimeDelay()` 是优化器、视觉缓存和视觉 FEJ 当前残差的入口。
- `time_delay` 是 `paramsToState()` 同步后的物理 current 值，供预测、三角化和跟踪快照使用；不能在优化器仍持有参数块时替代参数块。
- 首次存在有效帧时由 `captureLinearizedTimeDelay()` 冻结 `linearized_time_delay`，之后任何优化迭代都不能覆盖；非法值直接抛出异常。
- 视觉边缘化使用 `getLinearizedTimeDelay()`，并以 `b_f = r_current - J_f delta_current` 构造 FEJ 常量项；IMU 因子不包含时间延迟。
- 窗口迁移不会重置时间延迟 FEJ 身份，只有 `State::reset()` 才会清除它。

## 10. 修改阶段

### 阶段 0：删除零速检测

范围：

- Estimator；
- WindowOptimizer 接口和实现；
- 相关测试。

验证：

- 编译；
- WindowOptimizer 接口测试；
- bias 不再在 solve 后被覆盖。

### 阶段 1：Frame 对象模型

范围：

- 引入 Frame 基类和三个派生类型；
- active 与 retained 所有权；
- `frame_id` 生命周期；
- current 参数接口和标志。

暂不引入 FEJ 行为，保持数值结果不变。

验证：

- 对象类型和 ID 不变量；
- reset 和窗口复用；
- Ceres 参数地址在单次 solve 内稳定。

### 阶段 2：State 窗口和预测

范围：

- `max_frame_num/current_frame_count`；
- Estimator 原始数据入口；
- Frame posterior prediction；
- 显式 PreintegrationEdge。

验证：

- 每条边端点和时间区间；
- 新帧从上一 posterior 预测；
- 窗口移动后没有悬空端点。

### 阶段 3：Feature 双深度

范围：

- current/linearized depth；
- FEJ generation；
- 三角化和优化写 current；
- 换宿主创建新身份。

验证：

- current 优化不覆盖 linearized；
- 换宿主后 generation 改变；
- 新深度使用 current 位姿计算。

### 阶段 4：WindowOptimizer 只使用 current

范围：

- current getter；
- current VisualFrameCache；
- 显式 IMU edge；
- accepted 结果提交。

验证：

- 视觉和 IMU解析 Jacobian 数值微分；
- accepted/rejected cache；
- 优化结果只写 current。

### 阶段 5：Marginalizer 实现 FEJ 线性模型

范围：

- Frame/Feature/delay current 与 linearized getter；
- 私有 linearized visual cache；
- explicit variable layout；
- IMUBlock 和 LandmarkBlock。

验证：

- current/linearized 版本和 generation 一致；
- `J_f * delta_c + b_f == r_c`；
- Jacobian 行列映射；
- 深度 QR；
- IMU、视觉边缘化 Jacobian 与 linearized 数值微分一致。

### 阶段 6：冻结 prior 和 gauge

范围：

- 删除 prior recenter，实现严格等价的 gauge 坐标变换；
- 求解前 retained current gauge 快照；
- current、FEJ、prior 和 retained 镜像事务式规范；
- 删除 GaugeAnchor。

验证：

- prior 的物理 FEJ 身份和 `b` 多轮优化后保持不变；
- current 改变时 prior residual 按 `H * delta + b` 改变；
- 随机全局平移+yaw 不改变活动物理因子代价；
- 完整 gauge 坐标变换前后 prior residual 在数值精度内相等；
- yaw 奇异输入抛出异常；
- 不发生部分提交。

### 阶段 7：清理旧索引契约

范围：

- 删除 `kRetainedFrameIndex` 等特殊布局假设；
- 删除隐式 preintegrator 下标；
- 更新变量布局、测试和注释。

验证：

- 不同窗口容量；
- Normal/Key/Retained 替换路径；
- frame ID 和变量列映射完整。

## 11. 测试策略

只测试稳定、确定性的契约。

必须覆盖：

- Frame 标志状态机和非法读取；
- State 容量、ID 和对象所有权；
- PreintegrationEdge 端点与时间区间；
- Feature FEJ generation 和换宿主；
- current cache 版本失效；
- active factor 数值 Jacobian；
- marginalization Jacobian 在 linearized 点的数值微分；
- FEJ 常数项在 current 点重建 residual；
- LandmarkBlock 行列映射和 QR；
- prior 冻结；
- gauge cost 不变和奇异性；
- zero-speed 机制已经退出接口。

不测试：

- 数据集是否存在；
- 某段轨迹是否达到指定精度；
- 用户调参结果；
- ROS 运行环境；
- 前端统计阈值的经验效果。

上述内容通过明确数据集回放和部署 benchmark 人工验证。

## 12. 完成标准

重构完成必须同时满足：

- Estimator 不直接管理状态字段、FEJ 数据或预积分数组；
- State 拥有 active frames、retained frame 和显式 IMU edges；
- Frame/Feature 分别拥有 current 和 linearized 数据；
- 三角化与活动优化只使用 current；
- 边缘化 residual 使用 current、Jacobian 使用 linearized，并正确构造固定坐标常数项；
- 换宿主建立新的特征 FEJ 身份；
- prior 不重新定心；gauge 坐标变换保持 prior residual 不变；
- current、FEJ 表示、prior 世界系列和 retained 镜像协同规范；
- 每个新帧从上一 posterior current 预测，并创建独立 FEJ 身份；
- 零速检测和 bias 恢复逻辑不存在；
- 所有矩阵布局由显式 ID 映射决定；
- 数学不变量具有确定性测试。
