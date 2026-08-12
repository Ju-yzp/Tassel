# Schmidt-EKF 关键公式摘录

来源：Patrick Geneva, James Maley, Guoquan Huang, *An Efficient Schmidt-EKF for
3D Visual-Inertial SLAM*, CVPR 2019, arXiv:1903.08636。

本文只摘录与状态分割、Schmidt 更新和 Tassel 中 `ba` 保持机制直接相关的公式。
公式编号沿用原论文。

## 1. 状态与协方差分块

论文将完整状态分为 active state 和 Schmidt state：

\[
\mathbf{x}_k =
\begin{bmatrix}
\mathbf{x}_{A,k} \\
\mathbf{x}_{S,k}
\end{bmatrix},
\qquad
\mathbf{x}_{A,k} =
\begin{bmatrix}
\mathbf{x}_{I,k} \\
\mathbf{x}_{C,k}
\end{bmatrix}.
\tag{1}
\]

其中，\(\mathbf{x}_I\) 是当前 IMU 导航状态，\(\mathbf{x}_C\) 是滑窗中的克隆位姿，
\(\mathbf{x}_S\) 是被视为 nuisance parameters 的成熟地图点。对应协方差为

\[
\mathbf{P}_k =
\begin{bmatrix}
\mathbf{P}_{AA,k} & \mathbf{P}_{AS,k} \\
\mathbf{P}_{SA,k} & \mathbf{P}_{SS,k}
\end{bmatrix}.
\tag{5}
\]

这里 \(\mathbf{P}_{AS}\) 不能丢弃。它表达 Schmidt 状态的不确定性如何与 active
state 相关，是“均值不更新但信息仍参与估计”的关键。

## 2. 线性化观测

视觉观测在线性化点附近写成

\[
\mathbf{r}_k
= \mathbf{H}_k\widetilde{\mathbf{x}}_{k|k-1} + \mathbf{n}_k,
\qquad
\mathbf{H}_k =
\begin{bmatrix}
\mathbf{H}_{A,k} & \mathbf{H}_{S,k}
\end{bmatrix},
\tag{12--14}
\]

其中 \(\mathbf{r}_k\) 是观测残差，\(\widetilde{\mathbf{x}}\) 是状态误差，
\(\mathbf{n}_k\sim\mathcal N(\mathbf 0,\mathbf R_k)\)。

若特征仅作为 MSCKF 约束使用，论文将残差投影到特征 Jacobian
\(\mathbf H_f\) 的左零空间。令 \(\mathbf N^T\mathbf H_f=0\)，则

\[
\mathbf N^T\mathbf r_f
= \mathbf N^T\mathbf H_x\widetilde{\mathbf x}_{A,k|k-1}
+ \mathbf N^T\mathbf n_f,
\tag{17}
\]

即

\[
\mathbf r'_f
= \mathbf H'_x\widetilde{\mathbf x}_{A,k|k-1}+\mathbf n'_f,
\qquad
\mathbf R'_f=\mathbf N^T\mathbf R_f\mathbf N.
\tag{18}
\]

这说明变量消除可以改变参与求解的结构，而不是仅重新缩放残差。

## 3. Schmidt 增益

完整 Kalman 增益按 active/Schmidt 状态分块为

\[
\begin{bmatrix}
\mathbf K_{A,k} \\
\mathbf K_{S,k}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf P_{AA,k|k-1}\mathbf H_{A,k}^T
+\mathbf P_{AS,k|k-1}\mathbf H_{S,k}^T \\
\mathbf P_{SA,k|k-1}\mathbf H_{A,k}^T
+\mathbf P_{SS,k|k-1}\mathbf H_{S,k}^T
\end{bmatrix}
\mathbf S_k^{-1}.
\tag{19}
\]

创新协方差是

\[
\mathbf S_k =
\begin{bmatrix}
\mathbf H_{A,k} & \mathbf H_{S,k}
\end{bmatrix}
\mathbf P_{k|k-1}
\begin{bmatrix}
\mathbf H_{A,k} & \mathbf H_{S,k}
\end{bmatrix}^{T}
+\mathbf R_k.
\tag{20}
\]

Schmidt 更新主动设置

\[
\mathbf K_{S,k}=\mathbf 0,
\]

但不会令 \(\mathbf H_S\)、\(\mathbf P_{AS}\) 或 \(\mathbf P_{SS}\) 为零。因此
Schmidt 状态的不确定性仍进入 \(\mathbf S_k\)，并通过 \(\mathbf K_A\) 影响 active
state 的更新。

## 4. 状态均值更新

Schmidt 更新只修改 active state：

\[
\widehat{\mathbf x}_{A,k|k}
=\widehat{\mathbf x}_{A,k|k-1}+\mathbf K_{A,k}\mathbf r_k,
\qquad
\widehat{\mathbf x}_{S,k|k}
=\widehat{\mathbf x}_{S,k|k-1}.
\tag{21}
\]

第二式就是严格意义上的“该变量不更新”。它并不等价于假设该变量完全准确。

## 5. 协方差更新

active state 的协方差更新为

\[
\mathbf P_{AA,k|k}
=\mathbf P_{AA,k|k-1}
-\mathbf K_{A,k}
\left(
\mathbf H_{A,k}\mathbf P_{AA,k|k-1}
+\mathbf H_{S,k}\mathbf P_{SA,k|k-1}
\right).
\tag{22}
\]

active/Schmidt 交叉协方差仍然更新：

\[
\mathbf P_{AS,k|k}
=\mathbf P_{AS,k|k-1}
-\mathbf K_{A,k}
\left(
\mathbf H_{A,k}\mathbf P_{AS,k|k-1}
+\mathbf H_{S,k}\mathbf P_{SS,k|k-1}
\right).
\tag{23}
\]

Schmidt 状态的边缘协方差保持不变：

\[
\mathbf P_{SS,k|k}=\mathbf P_{SS,k|k-1}.
\tag{24}
\]

因此，Schmidt 状态同时满足：均值不更新、自身边缘协方差不更新、与 active
state 的交叉协方差继续更新。

## 6. 与 Tassel 中 `ba` 门控的对应

论文把成熟地图点作为 Schmidt state；Tassel 的候选机制则在低激励阶段把
加速度计偏置 `ba` 暂时作为 Schmidt/consider state。在线性化的一次更新内，可写为

\[
\delta\mathbf x=
\begin{bmatrix}
\delta\mathbf x_A \\
\delta\mathbf b_a
\end{bmatrix},
\qquad
\delta\mathbf b_a=\mathbf 0.
\]

这里的对应关系是：

- \(\mathbf x_A\)：仍允许更新的位姿、速度、陀螺仪偏置等状态；
- \(\mathbf x_S\)：当前被冻结的 `ba`；
- \(\mathbf P_{AS}\)：`ba` 与 active state 的交叉不确定性；
- \(\mathbf H_S\)：观测或先验对 `ba` 的 Jacobian。

仅在 Ceres 中把 `ba` 设为常量，只能保证 \(\delta\mathbf b_a=0\)。若同时丢弃
\(\mathbf P_{AS}\)、\(\mathbf P_{SS}\) 或等价的平方根先验信息，就不再等价于论文的
Schmidt 更新。Tassel 的平方根先验实现需要维持上述线性高斯更新所表达的信息关系。

## 7. 计算复杂度结论

论文指出，标准 EKF-SLAM 对地图规模 \(n\) 的协方差更新通常具有
\(\mathcal O(n^2)\) 复杂度。Schmidt state 的均值和 \(\mathbf P_{SS}\) 不参与更新，
而只维护 active/Schmidt 交叉项，使传播、更新与状态管理在每次只使用有界数量地图点
时降为 \(\mathcal O(n)\)。这是论文采用 Schmidt 分割的原始动机；Tassel 使用该分割
控制弱可观变量时，首要动机则是信息一致性，而不是地图规模复杂度。
