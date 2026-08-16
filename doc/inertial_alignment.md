# 单目 SFM 与惯性对齐

本文推导 Tassel 在单目 SFM 之后执行的惯性对齐机制。目标是利用 IMU 预积分恢复陀螺仪
偏置、公制尺度、重力方向和各帧速度，并把无尺度视觉轨迹转换为 VIO 初始状态。

## 1. 初始化顺序

`DynamicInitializer::initialize()` 当前严格按以下顺序执行：

```text
无尺度 SFM 轨迹
  -> 视觉相对旋转 + IMU 预积分求陀螺仪偏置
  -> 使用新陀螺仪偏置重新传播全部预积分
  -> 线性求速度、重力和尺度
  -> 固定重力模长，在二维切空间迭代精化
  -> 将重力旋转到世界 +Z 并消除初始 yaw
  -> 写入 R/P/V/Bg，进入正常 VIO 优化
```

这是顺序初始化，而不是在 SFM 因子中同时联合优化所有惯性状态。

## 2. 坐标系和视觉输出

[SFM 初始化](sfm_initialization.md)输出：

- $\mathbf R_k^v$：相机 $k$ 到视觉参考系 $\mathcal V$ 的旋转；
- $\mathbf p_{C_k}^v$：相机中心在 $\mathcal V$ 中的无尺度位置；
- 第 0 帧满足 $\mathbf R_0^v=\mathbf I$、$\mathbf p_{C_0}^v=\mathbf0$。

外参采用

```math
\mathbf p_I=\mathbf R_{IC}\mathbf p_C+\mathbf t_{IC},
```

其中 `ric = R_IC`、`tic = t_IC`。定义中间参考系 $\mathcal F$ 与第 0 帧 IMU 轴对齐，
则第 $k$ 帧 IMU 姿态为

```math
\mathbf R_k^F
=\mathbf R_{IC}\mathbf R_k^v\mathbf R_{IC}^T.
```

考虑相机和 IMU 的杆臂后，IMU 位置为

```math
\boxed{
\mathbf p_{I_k}^F
=s\mathbf R_{IC}\mathbf p_{C_k}^v
-\mathbf R_k^F\mathbf t_{IC}
+\mathbf t_{IC}
}.
```

这里 $s$ 是待估计公制尺度。最后一项使第 0 帧 IMU 原点位于 $\mathcal F$ 原点。

## 3. IMU 预积分运动方程

Tassel 使用正模长重力向量 $\mathbf G$，运行时平动动力学为

```math
\dot{\mathbf v}=\mathbf R\mathbf a-\mathbf G.
```

对相邻图像帧 $i,j$，预积分量
$\Delta\mathbf R_{ij}$、$\Delta\mathbf v_{ij}$、$\Delta\mathbf p_{ij}$ 表达在
第 $i$ 帧 IMU 坐标系。离散运动方程为

```math
\mathbf R_j^F
=\mathbf R_i^F\Delta\mathbf R_{ij},
```

```math
\mathbf v_j^F
=\mathbf v_i^F-\mathbf G^F\Delta t_{ij}
+\mathbf R_i^F\Delta\mathbf v_{ij},
```

```math
\mathbf p_{I_j}^F
=\mathbf p_{I_i}^F+\mathbf v_i^F\Delta t_{ij}
-\frac{1}{2}\mathbf G^F\Delta t_{ij}^2
+\mathbf R_i^F\Delta\mathbf p_{ij}.
```

符号中的负重力项与运行时 `R * acc - G` 保持一致。

## 4. 陀螺仪偏置求解

视觉轨迹给出的相邻 IMU 相对旋转为

```math
\mathbf R_{ij}^{\mathrm{vis}}
=\left(\mathbf R_i^F\right)^T\mathbf R_j^F
=\mathbf R_{IC}(\mathbf R_i^v)^T
\mathbf R_j^v\mathbf R_{IC}^T.
```

在当前预积分偏置线性化点附近，旋转预积分的一阶偏置修正为

```math
\Delta\mathbf R_{ij}(\mathbf b_g+\delta\mathbf b_g)
\simeq
\Delta\mathbf R_{ij}(\mathbf b_g)
\operatorname{Exp}
\left([\mathbf J_{ij}^{R,b_g}\delta\mathbf b_g]_\times\right).
```

定义视觉与预积分旋转差：

```math
\boldsymbol\phi_{ij}
=\operatorname{Log}
\left(
\Delta\mathbf R_{ij}^T\mathbf R_{ij}^{\mathrm{vis}}
\right).
```

一阶线性关系为

```math
\mathbf J_{ij}^{R,b_g}\delta\mathbf b_g
\simeq\boldsymbol\phi_{ij}.
```

堆叠全部相邻帧并直接对 Jacobian 求最小二乘：

```math
\boxed{
\delta\mathbf b_g
=\mathop{\arg\min}_{\delta\mathbf b}
\left\|\mathbf J\delta\mathbf b-\boldsymbol\phi\right\|_2
}.
```

窗口内预积分必须共享同一个偏置线性化点 $\bar{\mathbf b}_g$。求解结果是修正量，最终偏置为

```math
\mathbf b_g=\bar{\mathbf b}_g+\delta\mathbf b_g.
```

最终偏置写入全部初始化帧，随后所有预积分重新传播。必须先重传播再做尺度和重力对齐，
否则 $\Delta\mathbf v$、$\Delta\mathbf p$ 仍对应旧的旋转偏置线性化点。

## 5. 速度、重力和尺度线性系统

定义

```math
\mathbf Q_i=(\mathbf R_i^F)^T
=\mathbf R_{IC}(\mathbf R_i^v)^T\mathbf R_{IC}^T,
```

```math
\mathbf H_i=\mathbf R_{IC}(\mathbf R_i^v)^T,
\qquad
\mathbf C_{ij}=\mathbf Q_i\mathbf R_j^F.
```

将视觉 IMU 位置代入预积分位置方程，并左乘 $\mathbf Q_i$，得到

```math
-\Delta t_{ij}\mathbf Q_i\mathbf v_i^F
+\frac{1}{2}\Delta t_{ij}^2\mathbf Q_i\mathbf G^F
+s\mathbf H_i(\mathbf p_{C_j}^v-\mathbf p_{C_i}^v)
=\Delta\mathbf p_{ij}
+\mathbf C_{ij}\mathbf t_{IC}
-\mathbf t_{IC}.
```

速度方程左乘 $\mathbf Q_i$ 后为

```math
-\mathbf Q_i\mathbf v_i^F
+\mathbf Q_i\mathbf v_j^F
+\Delta t_{ij}\mathbf Q_i\mathbf G^F
=\Delta\mathbf v_{ij}.
```

未知量堆叠为

```math
\mathbf x=
\begin{bmatrix}
(\mathbf v_0^F)^T&\cdots&(\mathbf v_{N-1}^F)^T&
(\mathbf G^F)^T&s'
\end{bmatrix}^T,
\qquad s'=100s.
```

尺度使用 $s'=100s$ 只是变量缩放，因此矩阵中的视觉位移列除以 100。每个相邻帧对
提供 6 条线性方程，堆叠成全局 Jacobian 后直接使用 SVD 求解：

```math
\boxed{
\mathbf x=\mathop{\arg\min}_{\mathbf y}
\|\mathbf A\mathbf y-\mathbf b\|_2
}.
```

实现不构造 $\mathbf A^T\mathbf A$，避免平方条件数。求解前要求方程数不少于未知数，并
检查奇异值条件数；线性结果还必须满足尺度为正、重力有限，且重力模长与配置值之差
不超过阈值。

## 6. 外参常数项

按第 2 节固定的外参约定，从预积分位置方程严格推导得到的常数项是

```math
\boxed{
\mathbf C_{ij}\mathbf t_{IC}-\mathbf t_{IC}
}.
```

`linearAlignment()` 和 `refineGravitySpeeds()` 均按该常数项组装。无噪声合成测试使用
非单位 `ric` 和非零 `tic`，能够恢复真值尺度、重力和各帧速度，验证其与最终状态写回的
外参约定一致。

## 7. 固定重力模长的切空间精化

线性求解把重力作为自由三维向量，不能严格保证 $\|\mathbf G\|=g_0$。精化阶段固定
重力模长，只优化球面切空间中的两个自由度。

令当前单位重力方向为

```math
\hat{\mathbf g}=\frac{\mathbf G}{\|\mathbf G\|}.
```

选择与 $\hat{\mathbf g}$ 正交的两个单位基向量，组成

```math
\mathbf T=
\begin{bmatrix}
\mathbf b_1^T\\
\mathbf b_2^T
\end{bmatrix},
\qquad
\mathbf T\hat{\mathbf g}=\mathbf0.
```

在当前方向附近参数化重力：

```math
\mathbf G
\simeq
\mathbf G_0+\mathbf L\mathbf w,
\qquad
\mathbf G_0=g_0\hat{\mathbf g},
\qquad
\mathbf L=g_0\mathbf T^T,
\qquad
\mathbf w\in\mathbb R^2.
```

把该式代入位置和速度方程，未知量变为

```math
\mathbf x_r=
\begin{bmatrix}
(\mathbf v_0^F)^T&\cdots&(\mathbf v_{N-1}^F)^T&
\mathbf w^T&s'
\end{bmatrix}^T.
```

每次线性求解后更新并重新归一化方向：

```math
\hat{\mathbf g}\leftarrow
\frac{\hat{\mathbf g}+\mathbf T^T\mathbf w}
{\left\|\hat{\mathbf g}+\mathbf T^T\mathbf w\right\|_2}.
```

当前实现固定迭代 4 次，最终取

```math
\mathbf G^F=g_0\hat{\mathbf g}.
```

## 8. 世界规范对齐和状态写回

惯性对齐仍保留绕重力轴的全局 yaw 自由度。代码先构造旋转 $\mathbf R_0$，使

```math
\mathbf R_0\frac{\mathbf G^F}{\|\mathbf G^F\|}
=\mathbf e_z,
```

再移除该旋转自身的 yaw，使初始化选择确定的航向规范。运行时重力被设置为

```math
\mathbf G=[0,0,g_0]^T.
```

最终状态为

```math
\boxed{
\mathbf R_k
=\mathbf R_0\mathbf R_{IC}\mathbf R_k^v\mathbf R_{IC}^T
},
```

```math
\boxed{
\mathbf P_k
=\mathbf R_0
\left(
s\mathbf R_{IC}\mathbf p_{C_k}^v
-\mathbf R_{IC}\mathbf R_k^v\mathbf R_{IC}^T\mathbf t_{IC}
+\mathbf t_{IC}
\right)
},
```

```math
\boxed{
\mathbf V_k=\mathbf R_0\mathbf v_k^F
}.
```

## 9. 可观性和失败条件

惯性对齐需要窗口内存在足够的旋转和平移激励：

- 角速度变化不足时，陀螺仪偏置与视觉旋转误差难以区分。
- 匀速或近似静止运动下，尺度、重力和速度可能高度相关。
- 近似恒定加速度会削弱尺度与重力方向的可分辨性。
- SFM 平移方向错误时，线性系统可能给出负尺度或错误重力。
- 外参平移较小时，其坐标项错误不容易在普通数据集上暴露，需要专门合成测试。

代码直接检查堆叠 Jacobian 的行列数、有限性和奇异值条件数，并继续检查正尺度和重力
模长。两帧欠定窗口和静止退化窗口会被明确拒绝；条件数阈值仍是工程判据，不能替代对
实际运动激励的检查。

## 10. 代码对应关系

| 机制 | 实现位置 |
| --- | --- |
| 初始化控制流 | `DynamicInitializer::initialize()` |
| 陀螺仪偏置修正量最小二乘 | `solveGyroBiasCorrection()` |
| 偏置重传播 | `MidPointIntegrator::repropagate()` |
| 速度、重力、尺度线性解 | `linearAlignment()` |
| 重力二维切空间精化 | `refineGravitySpeeds()` |
| 世界规范和状态写回 | `DynamicInitializer::initialize()` |
