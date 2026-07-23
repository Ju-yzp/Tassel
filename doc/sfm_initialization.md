# 单目 SFM 初始化机制

本文说明 Tassel 的 `InitialSFM` 如何从滑窗内的连续特征观测恢复无尺度相机轨迹和
路标。文档只描述当前代码实际执行的机制；陀螺仪偏置、尺度、重力和速度由后续
[惯性对齐](inertial_alignment.md) 顺序求解，不属于 SFM 内部变量。

## 1. 输入、输出与尺度自由度

输入来自 `FeatureManager::collectSFMFeatures()`。每条特征包含归一化相机观测
$\mathbf u_{lk}=[u_{lk},v_{lk}]^T$ 及其帧索引 $k$。当前实现收集活动窗口内的全部连续
观测，越界索引视为程序错误。

SFM 输出：

- 相机到视觉参考系的旋转 $\mathbf R_k^v$；
- 相机中心在视觉参考系中的位置 $\mathbf p_k^v$；
- 通过正深度检查的无尺度三维路标 $\mathbf X_l^v$。

单目几何只能恢复平移方向，不能恢复公制尺度。若

```math
\left\{\mathbf p_k^v,\mathbf X_l^v\right\}
```

满足所有投影约束，则任意 $\lambda>0$ 下

```math
\left\{\lambda\mathbf p_k^v,\lambda\mathbf X_l^v\right\}
```

产生相同的归一化图像观测。当前实现将种子帧与另一帧之间的平移方向归一化，从而选择
一个任意视觉尺度；公制尺度 $s$ 留给惯性对齐恢复。

## 2. 初始化流程

当前流程为：

```text
连续归一化观测
  -> 选择连接稳定的种子帧
  -> 选择高视差候选帧（最多两个）
  -> Essential RANSAC
  -> 分解 E 并用正深度消除四解歧义
  -> 两视图三角化
  -> PnP 注册其余帧并继续三角化
  -> 全窗口 Sampson 对极优化
  -> 多视图重新三角化和正深度检查
  -> 对齐到第 0 帧参考系
```

任一候选失败时，系统尝试下一个候选；全部失败后，本次 VIO 初始化结束并等待后续图像。

## 3. 种子帧和视差帧选择

### 3.1 种子帧

对每个帧 $k$，代码统计观测特征数 $N_k$，并对同一轨迹在邻近三帧内的连接赋权：

```math
S_k=\sum_{l}\sum_{m\in\mathcal O_l}
\begin{cases}
4-|k-m|, & 0<|k-m|\le 3,\\
0, & \text{其他情况}.
\end{cases}
```

其中 $\mathcal O_l$ 是特征 $l$ 的观测帧集合。只考虑满足
$N_k\ge N_{\mathrm{seed,min}}$ 的帧，并选择 $S_k$ 最大者作为种子帧。该准则优先选择
位于连续长轨迹中心、与邻近帧连接稳定的图像，而不是固定选择窗口首帧或末帧。

### 3.2 另一帧

对种子帧 $s$ 和候选帧 $j$，收集共有特征并计算归一化平面视差：

```math
d_l^{sj}=\left\|\mathbf u_{ls}-\mathbf u_{lj}\right\|_2.
```

使用中位数

```math
\bar d_{sj}=\operatorname{median}_l d_l^{sj}
```

抑制少量错误跟踪。候选首先按中位视差降序、再按共有点数降序排列，最多尝试两个。
若所有候选都低于视差阈值，但仍存在足够视觉连接，当前实现退化为选择时间距离更远且
共有点更多的帧。该退化能增加初始化机会，但不能消除纯旋转或极低基线的退化。

## 4. Essential 矩阵与四解歧义

归一化齐次观测记为

```math
\mathbf x_s=[u_s,v_s,1]^T,
\qquad
\mathbf x_j=[u_j,v_j,1]^T.
```

若点从种子相机到候选相机满足

```math
\mathbf p_j=\mathbf R_{js}\mathbf p_s+\mathbf t_{js},
```

则 Essential 矩阵为

```math
\mathbf E=[\mathbf t_{js}]_\times\mathbf R_{js},
```

对极约束为

```math
\boxed{
\mathbf x_j^T\mathbf E\mathbf x_s=0
}.
```

代码在归一化坐标上调用 `findEssentialMat()`，因此相机矩阵取单位阵，并使用 RANSAC
剔除不满足对极约束的匹配。

对 $\mathbf E=\mathbf U\mathbf\Sigma\mathbf V^T$，定义

```math
\mathbf W=
\begin{bmatrix}
0&-1&0\\
1&0&0\\
0&0&1
\end{bmatrix}.
```

候选解为

```math
\mathbf R_1=\mathbf U\mathbf W\mathbf V^T,
\qquad
\mathbf R_2=\mathbf U\mathbf W^T\mathbf V^T,
\qquad
\mathbf t=\pm\mathbf U_{:,3}.
```

这产生 $(\mathbf R_1,\pm\mathbf t)$ 和 $(\mathbf R_2,\pm\mathbf t)$ 四组候选。对每组
候选执行线性三角化，统计同时位于两台相机前方的点数：

```math
Z_s>z_{\min},
\qquad
\left(\mathbf R_{js}\mathbf X_s+\mathbf t_{js}\right)_z>z_{\min}.
```

正深度点最多的候选被接受。随后将相对平移转换为相机中心方向并归一化：

```math
\mathbf p_j^s=
\frac{-\mathbf R_{js}^T\mathbf t_{js}}
{\left\|-\mathbf R_{js}^T\mathbf t_{js}\right\|_2}.
```

## 5. DLT 三角化

设世界点齐次坐标为 $\tilde{\mathbf X}$，投影矩阵为
$\mathbf P_k=[\mathbf R_{C_kv}\mid\mathbf t_{C_kv}]$。归一化观测满足

```math
\tilde{\mathbf x}_k\times\mathbf P_k\tilde{\mathbf X}=0.
```

每个视图提供两条独立线性方程：

```math
\begin{bmatrix}
u_k\mathbf P_{k,3}^T-\mathbf P_{k,1}^T\\
v_k\mathbf P_{k,3}^T-\mathbf P_{k,2}^T
\end{bmatrix}
\tilde{\mathbf X}=0.
```

两视图时堆叠为 $4\times4$ 矩阵，多视图时堆叠为 $2m\times4$ 矩阵。使用 SVD 取最小
奇异值对应的右奇异向量，再进行齐次归一化：

```math
\mathbf X=\frac{1}{\tilde X_4}
\begin{bmatrix}\tilde X_1&\tilde X_2&\tilde X_3\end{bmatrix}^T.
```

非有限结果和负深度结果不会进入最终路标集合。

## 6. PnP 注册其余帧

种子帧和候选帧建立初始地图后，算法每次选择与已求解帧时间距离最近的未求解帧。
对该帧收集已有三维点与二维归一化观测，求解

```math
\mathbf u_{lk}\simeq
\pi\left(\mathbf R_{C_kv}\mathbf X_l^v+\mathbf t_{C_kv}\right).
```

PnP 使用最近已求解帧的位姿作为初值。求解后逐点检查归一化平面重投影误差：

```math
e_{lk}=\left\|
\pi\left(\mathbf R_{C_kv}\mathbf X_l^v+\mathbf t_{C_kv}\right)
-\mathbf u_{lk}
\right\|_2.
```

平均误差超阈值或坏点比例过高时，整帧注册失败。成功注册后，使用新帧与两端种子帧
继续三角化尚未初始化的轨迹。

## 7. 全窗口 Sampson 对极优化

当前优化不是传统 Bundle Adjustment。优化变量只有每帧世界到相机的旋转和平移：

```math
\mathcal X=
\left\{\mathbf R_{C_kv},\mathbf t_{C_kv}\right\}_{k=0}^{N-1}.
```

三维路标不参与该非线性优化。对同一轨迹的任意两次观测 $i,j$，定义

```math
\mathbf A_{ji}=\mathbf R_{C_jv}\mathbf R_{C_iv}^T,
\qquad
\mathbf t_{ji}=\mathbf t_{C_jv}-\mathbf A_{ji}\mathbf t_{C_iv}.
```

于是

```math
\mathbf E_{ji}=[\mathbf t_{ji}]_\times\mathbf A_{ji}.
```

令

```math
\mathbf e=\mathbf E_{ji}\mathbf x_i,
\qquad
\mathbf h=\mathbf E_{ji}^T\mathbf x_j,
\qquad
n=\mathbf x_j^T\mathbf E_{ji}\mathbf x_i,
```

代码使用一阶 Sampson 残差

```math
\boxed{
r_{ij}=
\frac{n}
{\sqrt{e_x^2+e_y^2+h_x^2+h_y^2+\epsilon}}
}.
```

所有有效轨迹的观测对构成稀疏对极边，并使用 Huber 损失抑制残余错误匹配。种子帧和
候选帧的旋转、平移都固定，以同时固定全局位姿规范和单目尺度规范。

优化完成后，轨迹长度不少于三帧的路标使用全部观测重新执行多视图 DLT，而不是继续
使用优化前的两视图深度。

## 8. 参考系对齐

SFM 内部首先以种子帧建立场景，但输出前统一变换到窗口第 0 帧坐标系。若原始输出为
$(\mathbf R_k^v,\mathbf p_k^v)$，则

```math
\mathbf R_k^{c_0}=\left(\mathbf R_0^v\right)^T\mathbf R_k^v,
```

```math
\mathbf p_k^{c_0}=\left(\mathbf R_0^v\right)^T
\left(\mathbf p_k^v-\mathbf p_0^v\right).
```

因此输出满足 $\mathbf R_0^{c_0}=\mathbf I$、$\mathbf p_0^{c_0}=\mathbf0$，并作为后续
视觉惯性对齐的视觉参考系。

## 9. 退化与失败机制

以下情况不能仅靠调低阈值解决：

- 纯旋转或近似纯旋转：平移方向和深度不可观。
- 基线过小：Essential 分解和三角化对噪声敏感。
- 近似平面场景：Essential 与单应模型可能竞争。
- 长轨迹不足：种子连接、PnP 和多视图三角化缺少约束。
- 大量动态点或错误跟踪：RANSAC、Huber 和正深度检查只能降低风险，不能保证正确。

SFM 成功只表示获得了几何一致的无尺度视觉轨迹，并不表示公制尺度、重力方向、速度和
IMU 偏置已经正确。这些量必须通过惯性对齐继续验证。

## 10. 代码对应关系

| 机制 | 实现位置 |
| --- | --- |
| 特征观测导出 | `FeatureManager::collectSFMFeatures()` |
| 种子帧选择 | `InitialSFM::selectSeedFrame()` |
| 视差候选排序 | `InitialSFM::findParallaxFrames()` |
| Essential RANSAC | `InitialSFM::computeEssential()` |
| 四解分解与正深度评分 | `decomposeEssentialMat()`、`scoreByCheirality()` |
| PnP 与 DLT | `registerFramePnP()`、`triangulateTwoFrames()` |
| Sampson 对极优化 | `EpipolarSampsonFactor`、`reconstructScene()` |
| 第 0 帧规范对齐 | `alignToReference()` |
