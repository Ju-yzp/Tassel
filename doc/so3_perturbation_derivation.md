# SO(3) 扰动与伴随变换推导

## 1. hat 与逆

- $a^\wedge b = a \times b$，即 $[a]_\times b$
- $a^\wedge = -(a^\wedge)^T$
- $\exp(\phi^\wedge)^T = \exp(-\phi^\wedge)$

## 2. 伴随恒等式（精确）

$R \in SO(3)$，$\phi \in \mathbb{R}^3$：

$$R \cdot \exp(\phi^\wedge) = \exp((R\phi)^\wedge) \cdot R$$

$$R^T \cdot \exp(\phi^\wedge) = \exp((R^T\phi)^\wedge) \cdot R^T$$

证明依赖 $R [v]_\times = [Rv]_\times R$（旋转保持叉积）。

## 3. 几何含义

| 扰动 | 旋转轴 |
|------|--------|
| $R \cdot \exp(\phi^\wedge)$（右扰动） | $\phi$ 在**体坐标系** |
| $\exp(\phi^\wedge) \cdot R$（左扰动） | $\phi$ 在**世界坐标系** |
| $R \cdot \phi_b = \phi_w$ | 体 $\to$ 世界 |
| $R^T \cdot \phi_w = \phi_b$ | 世界 $\to$ 体 |

伴随变换本质：**把旋转轴在体坐标系和世界坐标系之间转换，同时把扰动位置左右互换。**

## 4. $R^T$ 右扰动的泰勒展开

设右扰动 $R \leftarrow R \cdot \exp(\delta\phi^\wedge)$，$\delta\phi$ 在体坐标系：

$$R_{\text{new}}^T = \exp(\delta\phi^\wedge)^T \cdot R^T = \exp(-\delta\phi^\wedge) \cdot R^T$$

一阶泰勒 $\exp(-\delta\phi^\wedge) \approx I - \delta\phi^\wedge$：

$$R_{\text{new}}^T \approx R^T - \delta\phi^\wedge \cdot R^T = R^T + (R^T \cdot)^\wedge \cdot \delta\phi$$

对任意向量 $v$：

$$R_{\text{new}}^T \cdot v \approx R^T v - \delta\phi^\wedge \cdot R^T v = R^T v + (R^T v)^\wedge \cdot \delta\phi$$

$$\boxed{\frac{\partial(R^T v)}{\partial \delta\phi} = (R^T v)^\wedge}$$

### 直观理解

雅各比是 **hat(世界坐标系下的 target 点坐标)**，等价于该 3D 点对体坐标系扰动的叉积矩阵。代码里到处都是 `skewSymmetric(pts_imu_j)` 就是这个道理。

## 5. $R^T$ 左扰动的泰勒展开

设左扰动 $R \leftarrow \exp(\delta\phi^\wedge) \cdot R$，$\delta\phi$ 在世界坐标系：

$$R_{\text{new}}^T = R^T \cdot \exp(\delta\phi^\wedge)^T = R^T \cdot \exp(-\delta\phi^\wedge)$$

泰勒展开后：

$$R_{\text{new}}^T \cdot v \approx R^T v - R^T \cdot \delta\phi^\wedge \cdot v = R^T v + R^T \cdot v^\wedge \cdot \delta\phi$$

$$\boxed{\frac{\partial(R^T v)}{\partial \delta\phi} = R^T \cdot v^\wedge}$$

## 6. VINS-Mono / Tassel 代码对应

projection_factor.cpp 中：

- 对 target 位姿旋转求导（雅各比写为 $\text{ric}^T \cdot \text{hat}(pj\_in\_I)$），用的就是 §4 的结论
- `pj_in_I = R_j^T * (pi_w - P_j)`，对 $R_j$ 的体坐标系右扰动求导，得到 $\text{hat}(pj\_in\_I)$

visual_reprojection.h 中：

- `J_j.block<2,3>(0,0) = reduce * (ric^T * hat(pj_in_I))` — 完全一致

## 7. 伴随变换在边缘化中的应用

边缘化的先验因子需要计算 $\delta = \text{Log}(R_{\text{lin}}^{-1} \cdot R)$，这是把当前估计 $R = R_{\text{lin}} \cdot \exp(\delta\phi^\wedge)$ 的体坐标系增量提取出来：

$$R_{\text{lin}}^{-1} \cdot R = \exp(\delta\phi^\wedge)$$

$$\delta\phi = \text{Log}(R_{\text{lin}}^{-1} \cdot R)$$

雅各比直接用 H 的对应块，因为 $\partial(\text{Log}(R_{\text{lin}}^{-1} \cdot R \cdot \exp(\varepsilon^\wedge))) / \partial\varepsilon |_{\varepsilon=0} = I$。
