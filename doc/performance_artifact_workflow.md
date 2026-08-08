# 性能实验产物管理

本文档约束 VIO 后端性能实验的可执行文件归档和 profiler 函数绑定流程。

## 目标

- 每次关键改动后保留对应的可执行文件，避免重复切分支和重编译。
- 可执行文件名称必须绑定实验内容、Tassel commit、Ceres commit 和生成时间。
- profiler 结论必须绑定到具体函数、源码文件和工程含义，避免只说“内存访问”或“线性求解慢”。

## 可执行文件归档

使用：

```bash
scripts/archive_benchmark_binary.sh \
  --build-dir build-vtune \
  --target tassel_core/test_euroc \
  --tag schur-plan-baseline \
  --note "Before Schur layout-plan cache"
```

产物目录默认位于：

```text
artifacts/binaries/
```

目录名格式：

```text
test_euroc__<tag>__tassel-<commit>__ceres-<commit>__<timestamp>
```

每个目录包含：

- `bin/`：目标可执行文件。
- `lib/`：Tassel 项目内共享库。
- `run.sh`：优先加载归档内共享库的运行入口。
- `README.md`：实验 tag、说明、commit、动态链接、CMake 配置和 git 状态。
- `meta/`：`sha256`、`ldd`、`readelf`、CMake cache 和当前 diff。

如果归档目录中的 `README.md` 显示工作区存在未提交 diff，必须在实验记录中说明该 diff 是否属于实验内容。

## 本机指令集构建

使用固定 VIO Schur 特化进行本机性能测试时，显式开启：

```bash
cmake -S . -B build-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DTASSEL_ENABLE_NATIVE_ARCH=ON
cmake --build build-native --target test_euroc -j5
```

该选项对 Tassel 和 vendored Ceres 的本地 C++ 目标统一添加 `-march=native`，默认关闭。开启后生成的二进制依赖构建机器指令集，不能作为通用发布产物。

在 Intel Core i5-13500H 上，以普通 `-O3` 为基线，对 EuRoC MH_01_easy、70 Hz 回放、单线程执行三次交替顺序 A/B：

| 指标 | 普通 O3 | 本机指令集 | 变化 |
|---|---:|---:|---:|
| Jacobian | 1.612 ms | 1.458 ms | -9.56% |
| Linear solver | 2.427 ms | 2.040 ms | -15.94% |
| Ceres total | 5.057 ms | 4.555 ms | -9.94% |
| Estimator | 7.001 ms | 6.137 ms | -12.34% |

两端每次运行均包含 3658 次 Ceres 调用。反汇编确认 `SchurEliminator<2,1,Dynamic>::ChunkOuterProduct` 从 XMM `mulpd` 变为 YMM `vfnmadd213pd`，因此收益覆盖 Schur 热点，而非只来自非关键代码。

## 函数绑定

VTune hotspots CSV 生成后，使用：

```bash
scripts/bind_profile_hotspots.py \
  /tmp/tassel-vtune-hotspots.csv \
  --limit 50 \
  --out /tmp/tassel-vtune-hotspot-bindings.md
```

绑定表会输出：

- 函数名。
- 模块名。
- 源码文件。
- 工程分类。
- 当前函数在 VIO/Ceres 管线中的含义。
- 下一步应该验证的方向。

函数绑定只用于定位优化方向，不替代源码审查。涉及数学公式或布局复用时，仍需要回到具体代码确认：

- 残差维度。
- block size。
- 参数块顺序。
- landmark/pose/td 的 Schur 分组。
- reduced Hessian 的 block 写入位置。
- 布局签名是否变化。

## 推荐实验顺序

1. 归档当前可执行版本。
2. 跑一次普通 EuRoC，记录 Ceres summary、Timing pipeline、ATE。
3. 跑 VTune hotspots，生成 CSV。
4. 用 `bind_profile_hotspots.py` 生成函数绑定表。
5. 只针对绑定表中的前几个后端热点做小改动。
6. 编译新版本并归档。
7. 用归档的两个可执行文件做 A/B，不再反复切换源码重新编译。

## 当前后端重点

现有 hotspots 已显示后端优先级：

```text
SchurEliminator<2,1,-1>::ChunkOuterProduct
MTM_mat1x4
MMM_mat1x4
MatrixMatrixMultiplyNaive
BlockRandomAccessDenseMatrix::GetCell
ReprojectionFactor::evaluateCached
```

这说明当前后端优化应优先绑定到 Schur layout plan、小矩阵乘法路径和 reduced Hessian block 写入位置，而不是继续泛化讨论“权重调整”或“内存访问”。
