# marginalizeSqrtToSqrt 详解

## 1. 背景：为什么要用 sqrt 形式

VINS-Mono 的边缘化分两步：先构造 $H = J^T J$（信息矩阵），再舒尔补。这相当于对**法方程**操作，条件数是 $\kappa(J)^2$，数值精度差。

Tassel 直接在 **sqrt 信息矩阵**（即 QR 分解后的 $R$ 因子）上做边缘化，条件数是 $\kappa(J)$，数值更稳定。核心思路来自 Kaess et al. (2012) 的 iSAM2 和滑动窗口滤波中的 QR 边缘化。

### 两种形式对比

| | 法方程形式 (VINS-Mono) | sqrt 形式 (Tassel) |
|---|---|---|
| 存储 | $H = J^T J$, $b = J^T r$ | $R$, $d$（满足 $R^T R = H$, $R^T d = b$） |
| 条件数 | $\kappa(J)^2$ | $\kappa(J)$ |
| 边缘化 | 舒尔补 $H' = H_{rr} - H_{rm} H_{mm}^{-1} H_{mr}$ | QR 消元（本文） |

## 2. 输入与输出

```
输入:
  Q2Jp  (rows × (marg_size + keep_size))  上三角 R 矩阵（sqrt 信息矩阵）
  Q2r   (rows × 1)                        sqrt 残差

列布局: [ x_marg (丢弃) | x_keep (保留) ]

输出:
  marg_sqrt_H  sqrt 先验信息矩阵（仅关于 x_keep）
  marg_sqrt_b  sqrt 先验残差（仅关于 x_keep）
```

输入 `Q2Jp` 已经是 QR 分解后的上三角形式，即：

$$Q^T J = \begin{bmatrix} R \\ 0 \end{bmatrix}$$

等价于最小二乘问题 $\min \|J x + r\|^2$ 已经被正交变换为：

$$\min_x \left\| \begin{bmatrix} R_m & R_c \\ 0 & R_k \end{bmatrix} \begin{bmatrix} x_m \\ x_k \end{bmatrix} + \begin{bmatrix} d_1 \\ d_2 \end{bmatrix} \right\|^2$$

其中 $R_m$ 对应 marg 列，$R_k$ 对应 keep 列，$R_c$ 是交叉项。

## 3. Householder QR 消元过程

函数通过**逐列 Householder 反射**对 marg 列做进一步的 QR 分解，从 marg 列的正交补空间中提取只关于 keep 变量的约束。

### 3.1 循环结构

```
for i = 0 .. cols-1:                    // 从左到右扫描每一列
    if total_rank >= rows: break        // 行已用完

    对 Q2Jp.col(i) 的对角线以下部分构造 Householder 反射

    if β > threshold:                   // 列线性无关 → 保留此秩
        用该反射消掉右侧所有列和 Q2r
        total_rank++
    else:                               // 列线性相关 → 丢弃
        对角元置 0

    清零对角线以下

    if i == marg_size - 1:              // 记录 marg 块的秩
        marg_rank = total_rank
```

### 3.2 Householder 构造

```cpp
Q2Jp.col(i).tail(remainingRows).makeHouseholderInPlace(hCoeff, beta);
```

对列向量的尾部 `v` 构造 Householder 反射 $H = I - \beta v v^T$，使得 $H v = \|v\| e_1$。

参数：
- `hCoeff` — Householder 向量的归一化系数（存于 `v(0)` 位置）
- `beta` — 缩放因子；$\beta = 0$ 意味着 $v$ 已经是零向量，列线性相关

### 3.3 秩判定

```cpp
if (std::abs(beta) > rank_threshold)  // rank_threshold = sqrt(ε)
```

$\beta$ 反映列的范数（非零程度）。如果 $\beta \approx 0$，该列已被前面的列张成的空间完全覆盖，线性相关，跳过。

这就是 **rank-revealing QR** — 通过数值容差自动丢弃不可观维度（如全局 yaw 的零空间方向），不需要像 VINS-Mono 那样事后对 $A_{mm}$ 做特征值截断。

### 3.4 反射应用

```cpp
Q2Jp.bottomRightCorner(remainingRows, remainingCols)
    .applyHouseholderOnTheLeft(v, hCoeff, temp_data + i + 1);
Q2r.tail(remainingRows)
    .applyHouseholderOnTheLeft(v, hCoeff, temp_data + cols);
```

反射 $H$ 左乘到右下角块：消去第 i 列在当前行以下的元素对后续列的影响。同时作用到 Q2r 上保持一致。

`temp_data` 是工作区，Eigen 的 `applyHouseholderOnTheLeft` 需要一个临时缓存。

## 4. 数学原理图解

### 处理前（输入 Q2Jp）

```
    marg_size    keep_size
   ┌──────────┬──────────┐
   │  R 块     │  R 块     │
   │ (marg)   │ (cross)  │
   │          │          │
   └──────────┴──────────┘
   │  0       │  R 块     │
   │          │ (keep)   │
   └──────────┴──────────┘
```

这不一定是严格的块上三角 — marg 列和 keep 列可能交错有非零元。

### 逐列 QR 后

```
    marg_size    keep_size
   ┌──────────┬──────────┐  ← rows 0 .. marg_rank-1
   │  R1      │  R1_cross│     包含 marg 变量信息
   ├──────────┼──────────┤
   │  0       │  R2_keep │  ← rows marg_rank .. total_rank-1
   │          │          │     只含 keep 变量信息
   ├──────────┼──────────┤
   │  0       │  0       │  ← rows total_rank .. end
   └──────────┴──────────┘     零行（秩不足）
```

### 最小二乘解释

QR 操作后，最小二乘问题等价于：

$$\min_{x_m, x_k} \left\| R_1 x_m + R_{1c} x_k + d_1 \right\|^2 + \left\| R_{2k} x_k + d_2 \right\|^2$$

对任意 $x_k$，总可以选 $x_m$ 使第一项为零（$R_1$ 是上三角满秩）：

$$x_m^* = -R_1^{-1}(R_{1c} x_k + d_1)$$

代入后只剩：

$$\min_{x_k} \left\| R_{2k} x_k + d_2 \right\|^2$$

所以边缘化先验为：

$$\boxed{\text{marg\_sqrt\_H} = R_{2k}, \quad \text{marg\_sqrt\_b} = d_2}$$

## 5. 与 VINS-Mono 的对比

| 步骤 | VINS-Mono | Tassel (本函数) |
|------|-----------|-----------------|
| 形成系统 | $H = J^T J$, $b = J^T r$ | $R = Q^T J$（sqrt 形式） |
| 秩处理 | 对 $A_{mm}$ 特征分解，截断小特征值 | Householder $\beta$ 阈值，自动 rank-revealing |
| 消元 | $H' = H_{rr} - H_{rm} H_{mm}^{-1} H_{mr}$ | QR 消去 marg 列对 keep 列的影响 |
| 先验分解 | $J_{\text{prior}} = \sqrt{S} V^T$ | $R_{2k}$ 天然是 sqrt 形式 |
| 数值条件 | $\kappa(H) \approx \kappa(J)^2$ | $\kappa(R) \approx \kappa(J)$ |

## 6. 边界情况

```cpp
Eigen::Index keep_valid_rows = std::max(total_rank - marg_rank, Eigen::Index(1));
```

`std::max(..., 1)` 保证即使 marg 消去后没有剩余行（所有约束都涉及 marg 变量），也至少保留一行零行，避免先验因子变成 0 维。这对应信息矩阵全零的退化情况。

## 7. 后续使用

`marg_sqrt_H` 和 `marg_sqrt_b` 被传入 `MarginalizationPrior`，作为先验因子加入下一轮 Ceres 优化。在 `MarginalizationPrior::Evaluate()` 中：

$$r = \text{marg\_sqrt\_H} \cdot \delta x + \text{marg\_sqrt\_b}$$

$$J = \text{marg\_sqrt\_H}$$

先验残差直接以 sqrt 形式线性表达，不再需要分解。
