#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Binding:
    category: str
    role: str
    action: str


def bind_function(function: str, full_function: str, module: str, source: str) -> Binding:
    text = f"{function} {full_function} {module} {source}"
    if "SchurEliminator" in text:
        return Binding(
            "Ceres Schur",
            "Schur 消元后的 reduced system 构造，主要是 landmark 外积更新。",
            "优先检查 block 布局计划缓存、pose/td 混合块特化、outer-product 内核。",
        )
    if "BlockRandomAccess" in text or "GetCell" in text:
        return Binding(
            "Ceres block access",
            "定位 reduced Hessian 中的 block cell，并返回写入位置和 stride。",
            "检查是否能预计算 cell 指针、row/col/stride，避免在内层循环重复索引。",
        )
    if "small_blas" in text or "MatrixMatrixMultiply" in text or "MatrixTranspose" in text:
        return Binding(
            "Ceres small BLAS",
            "小块矩阵乘法，负责 JtJ、Jtr、Schur 外积中的固定尺寸乘加。",
            "优先做固定尺寸内核、混合 6/1 block 分派和 fused update。",
        )
    if "ReprojectionFactor" in text or "ResidualBlock::Evaluate" in text or "ProgramEvaluator" in text:
        return Binding(
            "Ceres residual/Jacobian",
            "残差和雅各比求值路径，包含视觉因子缓存读取和 Ceres residual block 调度。",
            "确认视觉缓存命中；若占比低，避免继续在残差公式上过度优化。",
        )
    if "cv::imread" in text or "cv::calcOpticalFlowPyrLK" in text or "cv::circle" in text:
        return Binding(
            "OpenCV front-end",
            "图像读取、LK 跟踪或可视化绘制。",
            "后端 profiling 时应隔离，避免前端/TBB 噪声污染 Ceres 结论。",
        )
    if "imencode" in text or "publish" in text or "libtassel_tools" in module:
        return Binding(
            "Visualization/ROS",
            "图像编码、发布或 viewer 相关路径。",
            "后端性能实验中应关闭或单独统计。",
        )
    if "libtbb" in module or "sched_yield" in function or "Stitch point" in function:
        return Binding(
            "Thread scheduling",
            "TBB/线程调度或 spin/yield 开销。",
            "确认来源是 OpenCV/ROS/可视化还是后端并行；Ceres 当前是 threads=1。",
        )
    if (
        "memcpy" in function
        or "memmove" in function
        or "operator new" in function
        or "free" in function
        or "resize" in function
        or "Mat::clone" in text
    ):
        return Binding(
            "Allocation/copy",
            "内存分配、释放、复制或容器扩容。",
            "结合 perf cache miss/IPC 判断是否真的构成 memory bottleneck。",
        )
    if "tassel_core" in module:
        return Binding(
            "Other Tassel core",
            "Tassel 后端或公共模块中的其它函数。",
            "需要结合调用栈进一步绑定到 estimator、marginalization 或 solver。",
        )
    return Binding(
        "Other/system",
        "系统库或未分类函数。",
        "除非占比高，否则不作为第一优化对象。",
    )


def parse_cpu_time(row: dict[str, str]) -> float:
    try:
        return float(row.get("CPU Time", "0") or "0")
    except ValueError:
        return 0.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        sample = csv_file.read(4096)
        csv_file.seek(0)
        dialect = csv.excel_tab if "\t" in sample else csv.excel
        return list(csv.DictReader(csv_file, dialect=dialect))


def write_markdown(rows: list[dict[str, str]], limit: int, output: Path) -> None:
    measured = [(parse_cpu_time(row), row) for row in rows]
    measured = [(time, row) for time, row in measured if time > 0.0]
    measured.sort(key=lambda item: item[0], reverse=True)
    total = sum(time for time, _ in measured)

    lines: list[str] = []
    lines.append("# Profile Hotspot Bindings")
    lines.append("")
    lines.append("该文件把 profiler 函数热点绑定到工程含义；分类只用于定位优化方向，不替代源码审查。")
    lines.append("")
    lines.append("| Rank | CPU Time | Share | Category | Function | Module | Source | Binding | Next check |")
    lines.append("|---:|---:|---:|---|---|---|---|---|---|")
    for rank, (time, row) in enumerate(measured[:limit], start=1):
        function = row.get("Function", "")
        full_function = row.get("Function (Full)", "")
        module = row.get("Module", "")
        source = row.get("Source File", "")
        binding = bind_function(function, full_function, module, source)
        share = 100.0 * time / total if total > 0.0 else 0.0
        lines.append(
            "| {rank} | {time:.3f}s | {share:.2f}% | {category} | `{function}` | `{module}` | `{source}` | {role} | {action} |".format(
                rank=rank,
                time=time,
                share=share,
                category=binding.category,
                function=function.replace("|", "\\|"),
                module=module.replace("|", "\\|"),
                source=source.replace("|", "\\|"),
                role=binding.role,
                action=binding.action,
            )
        )

    category_time: dict[str, float] = {}
    for time, row in measured:
        binding = bind_function(
            row.get("Function", ""),
            row.get("Function (Full)", ""),
            row.get("Module", ""),
            row.get("Source File", ""),
        )
        category_time[binding.category] = category_time.get(binding.category, 0.0) + time

    lines.append("")
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | CPU Time | Share |")
    lines.append("|---|---:|---:|")
    for category, time in sorted(category_time.items(), key=lambda item: item[1], reverse=True):
        share = 100.0 * time / total if total > 0.0 else 0.0
        lines.append(f"| {category} | {time:.3f}s | {share:.2f}% |")

    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind profiler hotspots to Tassel/Ceres roles.")
    parser.add_argument("csv", type=Path, help="VTune hotspots CSV file")
    parser.add_argument("--limit", type=int, default=40, help="Number of rows in the binding table")
    parser.add_argument("--out", type=Path, required=True, help="Markdown output path")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(rows, args.limit, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
