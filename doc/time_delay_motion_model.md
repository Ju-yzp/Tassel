# 相机-IMU 时间延迟运动补偿模型

本文推导 Tassel 当前使用的相机-IMU 时间延迟补偿模型，并将运动学公式与
`State::get_compensated_state()`、`ReprojectionFactor` 和通用重投影函数中的变量对应起来。

## 1. 坐标系与符号

使用以下坐标系：

- $\mathcal W$：VIO 世界坐标系。
- $\mathcal I_k$：第 $k$ 帧对应的 IMU 体坐标系。
- $\mathcal C_k$：第 $k$ 帧对应的相机坐标系。

状态与代码变量的对应关系为：

| 数学符号 | 代码变量 | 含义 |
| --- | --- | --- |
| $\mathbf R_k$ | `frames[k].R` | 从 $\mathcal I_k$ 到 $\mathcal W$ 的旋转 |
| $\mathbf P_k$ | `frames[k].P` | IMU 原点在 $\mathcal W$ 中的位置 |
| $\mathbf V_k$ | `frames[k].V` | IMU 在 $\mathcal W$ 中的速度 |
| $\mathbf b_{a,k}$ | `frames[k].Ba` | 加速度计偏置 |
| $\mathbf b_{g,k}$ | `frames[k].Bg` | 陀螺仪偏置 |
| $\mathbf R_{IC}$ | `ric` | 从相机系到 IMU 系的旋转 |
| $\mathbf t_{IC}$ | `tic` | 相机原点在 IMU 系中的位置 |
| $d$ | `delay_time` | 当前优化的全局相机-IMU 时间延迟 |
| $d_k^s$ | `sync_delay` | 第 $k$ 帧进入同步器时采用的延迟快照 |

当前优化延迟相对同步快照的剩余补偿量为

```math
\Delta t_k = d-d_k^s.
```

因此，$\Delta t_k$ 可以为正，也可以为负。正值表示从同步时刻向后传播，负值表示
向前外推。下文所有公式对有符号的 $\Delta t_k$ 都成立。

定义去偏后的体坐标系 IMU 测量：

```math
\boldsymbol\omega_k = \boldsymbol\omega_{m,k}-\mathbf b_{g,k},
\qquad
\mathbf a_k = \mathbf a_{m,k}-\mathbf b_{a,k}.
```

记 $[\mathbf x]_\times$ 为向量 $\mathbf x$ 的反对称矩阵，满足
$[\mathbf x]_\times\mathbf y=\mathbf x\times\mathbf y$。代码中的重力常量
`G = [0, 0, 9.8]^T`，因此世界系平动加速度采用

```math
\dot{\mathbf V}(t)=\mathbf R(t)\mathbf a_k-\mathbf G.
```

## 2. 局部运动假设

补偿区间通常远短于相邻图像间隔。当前模型在区间
$[t_k,t_k+\Delta t_k]$ 内采用以下局部假设：

1. 去偏角速度 $\boldsymbol\omega_k$ 在体坐标系中保持常量。
2. 去偏比力 $\mathbf a_k$ 在体坐标系中保持常量。
3. $\mathbf V_k$、$\mathbf b_{a,k}$ 和 $\mathbf b_{g,k}$ 是区间起点状态。
4. 重力 $\mathbf G$ 在世界系中保持常量。

该模型是局部运动近似，不是任意时变 IMU 输入下的精确积分。实现会在图像时刻附近对
IMU 测量进行插值，以减小常值假设的误差。

## 3. 姿态补偿

刚体姿态运动学为

```math
\dot{\mathbf R}(t)=\mathbf R(t)[\boldsymbol\omega_k]_\times.
```

由于 $\boldsymbol\omega_k$ 在补偿区间内为常量，该微分方程具有闭式解：

```math
\boxed{
\bar{\mathbf R}_k
=\mathbf R_k\operatorname{Exp}
\left([\boldsymbol\omega_k\Delta t_k]_\times\right)
}
```

定义

```math
\mathbf A_k=\operatorname{Exp}
\left([\boldsymbol\omega_k\Delta t_k]_\times\right),
```

则 $\bar{\mathbf R}_k=\mathbf R_k\mathbf A_k$。在角速度常值假设下，姿态补偿
关于 $\Delta t_k$ 是精确的。

## 4. 速度与位置补偿

世界系加速度为

```math
\ddot{\mathbf P}(t_k+\tau)
=\mathbf R_k\operatorname{Exp}
\left([\boldsymbol\omega_k\tau]_\times\right)\mathbf a_k-\mathbf G.
```

对指数映射在 $\tau=0$ 处展开：

```math
\operatorname{Exp}
\left([\boldsymbol\omega_k\tau]_\times\right)
=\mathbf I+[\boldsymbol\omega_k]_\times\tau+\mathcal O(\tau^2).
```

代入加速度方程：

```math
\ddot{\mathbf P}(t_k+\tau)
=\mathbf R_k\mathbf a_k-\mathbf G
+\mathbf R_k[\boldsymbol\omega_k]_\times\mathbf a_k\tau
+\mathcal O(\tau^2).
```

定义世界系旋转加速度修正项

```math
\mathbf c_k
=\mathbf R_k[\boldsymbol\omega_k]_\times\mathbf a_k.
```

对加速度积分一次，得到速度补偿：

```math
\boxed{
\bar{\mathbf V}_k
=\mathbf V_k
+(\mathbf R_k\mathbf a_k-\mathbf G)\Delta t_k
+\frac{1}{2}\mathbf c_k\Delta t_k^2
}
+\mathcal O(\Delta t_k^3).
```

再次积分，得到位置补偿：

```math
\boxed{
\bar{\mathbf P}_k
=\mathbf P_k
+\mathbf V_k\Delta t_k
+\frac{1}{2}(\mathbf R_k\mathbf a_k-\mathbf G)\Delta t_k^2
+\frac{1}{6}\mathbf c_k\Delta t_k^3
}
+\mathcal O(\Delta t_k^4).
```

这解释了实现中二阶速度项和三阶位置项的来源。三阶位置项并不是额外假设出的经验项，
而是体坐标系比力随刚体旋转进入世界系后，对时间积分两次得到的结果。

位置对延迟补偿量的导数为

```math
\boxed{
\frac{\partial\bar{\mathbf P}_k}{\partial\Delta t_k}
=\mathbf V_k
+(\mathbf R_k\mathbf a_k-\mathbf G)\Delta t_k
+\frac{1}{2}\mathbf c_k\Delta t_k^2
}.
```

由于 $\partial\Delta t_k/\partial d=1$，该式也直接用于全局延迟 $d$ 的视觉雅可比。

## 5. 宿主路标到目标相机的传播

设宿主帧为 $i$，目标帧为 $j$。宿主归一化齐次观测为
$\mathbf u_i=[u_i,v_i,1]^T$，逆深度为 $\rho$。宿主相机中的三维点为

```math
\mathbf p_{C_i}=\frac{1}{\rho}\mathbf u_i.
```

通过相机到 IMU 外参：

```math
\mathbf p_{I_i}=\mathbf R_{IC}\mathbf p_{C_i}+\mathbf t_{IC}.
```

使用补偿后的宿主状态变换到世界系：

```math
\boxed{
\mathbf p_W
=\mathbf R_i\mathbf A_i\mathbf p_{I_i}+\bar{\mathbf P}_i
}.
```

目标 IMU 中的点为

```math
\mathbf p_{I_j}
=\bar{\mathbf R}_j^T(\mathbf p_W-\bar{\mathbf P}_j)
=\mathbf A_j^T\mathbf R_j^T(\mathbf p_W-\bar{\mathbf P}_j).
```

再变换到目标相机：

```math
\boxed{
\mathbf p_{C_j}
=\mathbf R_{IC}^T(\mathbf p_{I_j}-\mathbf t_{IC})
}.
```

令 $\pi(\cdot)$ 表示透视除法、相机畸变和像素映射，$\mathbf z_j$ 为目标像素观测，
$\mathbf W^{1/2}$ 为视觉平方根信息矩阵，则重投影残差为

```math
\boxed{
\mathbf r_{ij}
=\mathbf W^{1/2}
\left(\pi(\mathbf p_{C_j})-\mathbf z_j\right)
}.
```

## 6. 视觉因子解析雅可比

定义投影链式雅可比

```math
\mathbf D
=\mathbf W^{1/2}
\frac{\partial\pi(\mathbf p_{C_j})}{\partial\mathbf p_{C_j}},
```

以及目标帧到宿主世界点的线性变换

```math
\mathbf M_j
=\mathbf R_{IC}^T\mathbf A_j^T\mathbf R_j^T.
```

### 6.1 平移雅可比

由 $\partial\mathbf p_W/\partial\mathbf P_i=\mathbf I$ 和
$\partial\mathbf p_{I_j}/\partial\mathbf P_j=-\mathbf A_j^T\mathbf R_j^T$，得到

```math
\boxed{
\frac{\partial\mathbf r_{ij}}{\partial\mathbf P_i}=\mathbf D\mathbf M_j,
\qquad
\frac{\partial\mathbf r_{ij}}{\partial\mathbf P_j}=-\mathbf D\mathbf M_j
}.
```

### 6.2 宿主姿态雅可比

定义体坐标系中的宿主旋转相关项

```math
\mathbf s_i
=\mathbf A_i\mathbf p_{I_i}
+\frac{1}{2}\mathbf a_i\Delta t_i^2
+\frac{1}{6}[\boldsymbol\omega_i]_\times\mathbf a_i\Delta t_i^3.
```

代码使用旋转向量 $\boldsymbol\phi_i$，且
$\mathbf R_i=\operatorname{Exp}([\boldsymbol\phi_i]_\times)$。对于加性旋转向量增量
$\delta\boldsymbol\phi_i$，有

```math
\mathbf R(\boldsymbol\phi_i+\delta\boldsymbol\phi_i)
\simeq
\mathbf R_i\operatorname{Exp}
\left([\mathbf J_l(-\boldsymbol\phi_i)\delta\boldsymbol\phi_i]_\times\right),
```

其中 $\mathbf J_l$ 是 $SO(3)$ 左雅可比。因此

```math
\boxed{
\frac{\partial\mathbf r_{ij}}{\partial\boldsymbol\phi_i}
=-\mathbf D\mathbf M_j\mathbf R_i
[\mathbf s_i]_\times\mathbf J_l(-\boldsymbol\phi_i)
}.
```

式中的最后一项

```math
\frac{1}{6}
[\boldsymbol\omega_i]_\times\mathbf a_i\Delta t_i^3
```

必须进入宿主姿态雅可比。忽略它在很小延迟下数值影响有限，但会使解析雅可比与三阶位置
补偿残差不一致。

### 6.3 目标姿态雅可比

将目标位置补偿代入 $\mathbf p_{I_j}$：

```math
\mathbf p_{I_j}
=\mathbf A_j^T
\left[
\mathbf R_j^T
\left(\mathbf p_W-\mathbf P_j-\mathbf V_j\Delta t_j
+\frac{1}{2}\mathbf G\Delta t_j^2\right)
-\frac{1}{2}\mathbf a_j\Delta t_j^2
-\frac{1}{6}[\boldsymbol\omega_j]_\times\mathbf a_j\Delta t_j^3
\right].
```

后两项已经位于目标 IMU 体坐标系，不再依赖 $\mathbf R_j$。定义

```math
\mathbf h_j
=\mathbf R_j^T
\left(\mathbf p_W-\mathbf P_j-\mathbf V_j\Delta t_j
+\frac{1}{2}\mathbf G\Delta t_j^2\right),
```

得到

```math
\boxed{
\frac{\partial\mathbf r_{ij}}{\partial\boldsymbol\phi_j}
=\mathbf D\mathbf R_{IC}^T\mathbf A_j^T
[\mathbf h_j]_\times\mathbf J_l(-\boldsymbol\phi_j)
}.
```

因此，目标姿态雅可比不需要额外添加三阶旋转加速度项；该项关于目标姿态的依赖在
$\mathbf R_j^T\bar{\mathbf P}_j$ 中已经抵消。

### 6.4 时间延迟雅可比

由矩阵指数的导数

```math
\frac{\partial(\mathbf A_i\mathbf p_{I_i})}{\partial\Delta t_i}
=\mathbf A_i[\boldsymbol\omega_i]_\times\mathbf p_{I_i},
```

以及

```math
\frac{\partial(\mathbf A_j^T\mathbf x)}{\partial\Delta t_j}
=-[\boldsymbol\omega_j]_\times\mathbf A_j^T\mathbf x,
```

定义

```math
\dot{\bar{\mathbf P}}_k
=\mathbf V_k
+(\mathbf R_k\mathbf a_k-\mathbf G)\Delta t_k
+\frac{1}{2}\mathbf R_k[\boldsymbol\omega_k]_\times\mathbf a_k\Delta t_k^2.
```

因为宿主和目标的 $\Delta t_i,\Delta t_j$ 对全局延迟 $d$ 的导数都为 1，故

```math
\frac{\partial\mathbf p_{I_j}}{\partial d}
=-[\boldsymbol\omega_j]_\times\mathbf p_{I_j}
+\mathbf A_j^T\mathbf R_j^T
\left(
\mathbf R_i\mathbf A_i[\boldsymbol\omega_i]_\times\mathbf p_{I_i}
+\dot{\bar{\mathbf P}}_i
-\dot{\bar{\mathbf P}}_j
\right).
```

最终得到

```math
\boxed{
\frac{\partial\mathbf r_{ij}}{\partial d}
=\mathbf D\mathbf R_{IC}^T
\frac{\partial\mathbf p_{I_j}}{\partial d}
}.
```

这里的 $\dot{\bar{\mathbf P}}_k$ 必须包含三阶位置项求导产生的二阶项，否则时间延迟
解析雅可比与残差不一致。

### 6.5 逆深度雅可比

由

```math
\frac{\partial\mathbf p_{C_i}}{\partial\rho}
=-\frac{\mathbf p_{C_i}}{\rho},
```

得到

```math
\boxed{
\frac{\partial\mathbf r_{ij}}{\partial\rho}
=-\mathbf D\mathbf R_{IC}^T\mathbf A_j^T\mathbf R_j^T
\mathbf R_i\mathbf A_i\mathbf R_{IC}
\frac{\mathbf p_{C_i}}{\rho}
}.
```

## 7. 优化与边缘化中的切空间

`ReprojectionFactor` 输出对旋转向量参数 $\boldsymbol\phi$ 的雅可比。Ceres 通过
`SE3RightManifold` 将其映射到局部右扰动。视觉因子写入平方根边缘化系统时，
`LandmarkBlock` 显式乘以相应的旋转映射，使视觉块、IMU 块和旧先验都表达在当前状态的
同一右扰动切空间中。

不能直接把 Basalt 中相对位姿到绝对位姿的雅可比缩放公式套用到这里。Tassel 当前视觉
因子直接连接宿主和目标的绝对状态，没有独立的相对位姿参数块，也不存在
$\mathbf J_{\mathrm{rel\to abs}}$ 这一层映射。

## 8. 适用范围与当前限制

当前解析雅可比对现有残差模型和四个参数块

```math
[\operatorname{pose}_i,\operatorname{pose}_j,d,\rho]
```

是一致的，但仍有以下模型边界：

1. 视觉因子构造时会保存 $\mathbf V_k$、$\mathbf b_{a,k}$、$\mathbf b_{g,k}$ 和
   IMU 测量快照；视觉残差不会直接对这些量求导。
2. 姿态在常角速度假设下使用精确指数映射；速度和位置来自局部泰勒展开。
3. 若补偿区间内存在明显角加速度、线加速度变化或 IMU 插值误差，模型误差会随
   $|\Delta t_k|$ 增大。
4. 若未来把速度和偏置加入视觉因子参数块，必须同时扩展残差雅可比、边缘化列布局、
   先验参数布局和数值雅可比测试。

## 9. 数值验证

`test_reprojection_factor` 使用 Ceres `GradientChecker` 和中心差分验证解析雅可比：

- 常规测试在真实量级的小延迟附近验证全部参数块。
- 高阶测试将延迟增大到 $0.2\,\mathrm{s}$，使二阶和三阶项具有足够数值幅度。
- 修正三阶导数遗漏前，大延迟测试检测到 8 个错误分量，最大相对误差约为
  $9.6\times10^{-3}$。
- 补齐宿主姿态和时间延迟雅可比后，数值微分测试通过。

该测试固定了本文公式与代码实现之间的数学契约。
