# Tassel VIO 后端重构设计

## 1. 文档目的

本文定义 Tassel 滑窗 VIO 后端的目标架构、数学契约、当前实现差距和分阶段迁移方案。

本次重构解决的核心问题不是管理层性能，而是：

- 状态、特征、预积分和 prior 的所有权不清晰；
- current、posterior 和 linearized 数据存在混用风险；
- 预积分端点依赖窗口下标；
- 当前边缘化并未统一使用 FEJ 数据；
- prior 在优化后被 recenter 和 gauge transform，违背目标 FEJ 定义；
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
- prior 不执行 recenter、tangent transport 或 gauge transform；
- current 窗口在优化结束后规范到保留帧的固定 linearized gauge。

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

唯一 gauge 参考是保留帧的 linearized 位姿：

```text
reference position = retained_frame.linearized_position
reference yaw      = yaw(retained_frame.linearized_rotation)
```

设保留帧 current 位姿为 `(R_c, p_c)`，冻结参考为 `(R_l, p_l)`：

```text
yaw_delta = yaw(R_l) - yaw(R_c)
R_g       = rotation_about_gravity(yaw_delta)
t_g       = p_l - R_g * p_c
```

优化后只变换 current 窗口，不变换 prior。

固定线性 prior 对 gauge 零空间的保持是一阶性质。对于有限且较大的 yaw 变换，流形上的 `current minus linearized` 不是全局线性函数，因此不能无条件宣称 prior 代价严格不变。工程约束是：

- 每轮优化提交后立即将 current 规范回固定 retained gauge；
- 不允许 gauge 漂移跨多轮累积；
- 小扰动测试验证平移/yaw 零空间；
- 有限变换测试验证物理活动因子代价不变，并验证 prior 误差符合预期的一阶模型，而不是强行要求机器精度完全相等。

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

生成后全部冻结。禁止：

- recenter prior 到 current；
- tangent transport 改写 `H`；
- gauge transform prior 的列或参考状态；
- 单独替换参考点、`H` 或 `b`；
- 依赖已经删除的历史变量重新线性化。

Ceres 所需的 local-to-ambient Jacobian 转换只能写入本次 `Evaluate()` 输出，不能写回 `H`。

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

窗口满时禁止覆盖对象。必须先结束当前 `Problem`、完成边缘化和窗口推进，再创建新帧。

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
3. 以 retained linearized pose 计算 gauge；
4. 规范全部 current pose 和 velocity；
5. 提交 current；
6. 标记 posterior；
7. 失效 current cache。

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
| gauge 规范 | current + retained linearized reference | 深度不变 | normalized current |

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

## 9. 当前实现差距

### 9.1 prior 被重新定心和变换

证据：

- `Estimator::optimize()` 调用 `MargHelper::recenterPrior()`；
- `normalizeGaugeAfterOptimization()` 调用 `transformPriorGauge()`。

问题：

- 改写 prior 的参考点、`H` 或 `b`；
- 与目标“冻结原始切空间”的 FEJ 定义冲突。

修改：

- 删除正常路径上的 recenter；
- 删除 gauge 对 prior 的变换；
- 只将 current 状态规范到 retained linearized gauge。

### 9.2 边缘化使用 current 数据

证据：

- `updateMarginalizationPrior()` 调用 `stateToParams()`；
- `LandmarkBlock` 读取 `param_pose`、`param_speed_bias` 和 `estimated_depth`；
- `MarginalizationSqrt` 使用 current `VisualFrameCache`；
- IMUBlock 从 current `FrameState` 字段线性化。

问题：

- 尚未实现目标 FEJ Jacobian；
- 没有构造固定 FEJ 坐标所需的 `b_f = r_c - J_f * delta_c`；
- 边缘化 Jacobian 和活动优化共享 current cache；
- 深度没有 linearized 版本。

修改：

- 为 Frame、Feature 和 delay 增加冻结 linearized 数据；
- Marginalizer 同时读取版本一致的 current/linearized getter；
- residual 使用 current，Jacobian 使用 linearized；
- 建立私有 linearized Jacobian cache，并构造 `b_f`。

### 9.3 旧 prior 依赖 recenter 后的 b

当前 reduced system 直接写入旧 prior 的 `H` 和 `b`。正常流程先 recenter，因此 `b` 已被改写到 current。

目标方案取消 recenter 后，旧 prior 必须与持续存在的 Frame linearized 身份完全一致。构造新 prior 时只能在同一组 per-variable FEJ 坐标中复用旧 `H/b`。若身份或 generation 不一致，应抛出错误，不能做隐式 transport。

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

### 9.9 独立 GaugeAnchor 会重复真相

当前 `State::gauge_reference` 单独保存 frame ID、旋转和位置。

问题：

- 它可能与 retained linearized pose 失配；
- 多一份 gauge 所有权。

修改：

- 删除独立 GaugeAnchor；
- retained linearized pose 是唯一参考。

### 9.10 零速检测不构成严格测量模型

当前实现以窗口速度阈值判断静止，然后在求解后恢复 accel bias 均值。

问题：

- 没有对应 residual、Jacobian、协方差和可观性分析；
- 求解器报告的 accepted 状态随后被人工修改；
- 优化模型和最终输出状态不一致。

修改：

- 当前重构直接删除该机制；
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

- 删除 recenter/transform prior；
- retained linearized gauge；
- current 事务式规范；
- 删除 GaugeAnchor。

验证：

- prior 数据多轮优化后逐字节或数值不变；
- current 改变时 prior residual 按 `H * delta + b` 改变；
- 随机全局平移+yaw 不改变活动物理因子代价；
- prior 的小 gauge 扰动满足一阶零空间，有限规范化误差受控且不累积；
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
- prior 生成后不再变化；
- gauge 只规范 current；
- 零速检测和 bias 恢复逻辑不存在；
- 所有矩阵布局由显式 ID 映射决定；
- 数学不变量具有确定性测试。
