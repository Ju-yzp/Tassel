#!/usr/bin/env bash
# ── Tassel profiling wrapper ────────────────────────────────────────────────
# Usage: ./profile.sh [--tool vtune|perf] [--mode <mode>] [--out <dir>] -- <program> [args...]
#
# Examples:
#   ./profile.sh -- ./build/tassel_core/test_lm_optimizer
#   ./profile.sh --tool perf --mode stat -- ./build/tassel_core/test_lm_optimizer
#   ./profile.sh --mode uarch --out ./prof_data -- ./build/tassel_core/test_feature
#
# VTune modes: hotspots (default), uarch, memory-access, memory-consumption,
#               threading, hpc-performance, io
# Perf modes:   record (default), stat, annotate, mem
#
# Long-running (I/O) programs: press Ctrl+C to stop profiling — results are
# saved automatically after interruption.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
TOOL="auto"
MODE="default"
OUT_DIR=""
INTERRUPTED=0

# ── colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    cat <<'USAGE'
── Tassel profiling wrapper ────────────────────────────────────────────────

Usage: ./profile.sh [--tool vtune|perf] [--mode <mode>] [--out <dir>] -- <program> [args...]

  --tool vtune|perf   Select profiling backend (default: auto-detect, prefer VTune)
  --mode <mode>       Analysis mode (see below)
  --out <dir>         Output directory for results

Examples:
  ./profile.sh -- ./build/tassel_core/test_lm_optimizer
  ./profile.sh --tool perf --mode stat -- ./build/my_binary --flag
  ./profile.sh --mode uarch --out ./prof_data -- ./build/test_feature

VTune modes:
  hotspots           采样热点分析 (默认)
  uarch              微架构探索 (需 root / perf_event_paranoid=0)
  memory-access       内存访问模式分析
  memory-consumption  内存消耗分析
  threading           线程并发分析
  hpc-performance     HPC 性能特征分析
  io                  I/O 分析

Perf modes:
  record             采样记录 (默认)
  stat               性能计数器统计
  mem                内存访问记录 (cache + 指令)
  annotate           源码级注释 (基于已有 perf.data)

Long-running programs: press Ctrl+C to stop — results are saved automatically.
USAGE
    exit 0
}

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $*"; }
log_cmd()  { echo -e "${CYAN}[CMD]${NC} $*"; }

# ── signal handler for long-running I/O programs ───────────────────────────
on_interrupt() {
    echo ""
    log_info "Interrupted — waiting for profiler to finalize results..."
    INTERRUPTED=1
}

# ── tool auto-detection ─────────────────────────────────────────────────────
detect_tool() {
    if [ -f /opt/intel/oneapi/vtune/latest/vtune-vars.sh ]; then
        echo "vtune"
    else
        echo "perf"
    fi
}

# ── binary sanity checks ────────────────────────────────────────────────────
check_binary() {
    local bin="$1"

    # Resolve the binary path (handles both "./build/foo" and "foo" in PATH)
    local resolved
    if [[ "$bin" == */* ]]; then
        resolved="$bin"
    else
        resolved=$(command -v "$bin" 2>/dev/null || true)
        if [ -z "$resolved" ]; then
            log_err "Program not found in PATH: $bin"
            exit 1
        fi
    fi

    if [ ! -f "$resolved" ] || [ ! -x "$resolved" ]; then
        log_err "Not an executable: $resolved"
        exit 1
    fi

    local file_info
    file_info=$(file "$resolved" 2>/dev/null || true)

    log_info "Binary: ${file_info}"

    # Check for debug symbols
    if command -v readelf &>/dev/null; then
        if readelf -S "$resolved" 2>/dev/null | grep -q '\.debug_info'; then
            log_info "Debug symbols: yes"
        else
            log_warn "No .debug_info section found — profiles may lack source-level detail"
            log_warn "Rebuild with: cmake -DCMAKE_BUILD_TYPE=Debug (or RelWithDebInfo)"
        fi
    fi

    # Check if stripped
    if echo "$file_info" | grep -q 'not stripped'; then
        log_info "Binary is not stripped — good for profiling"
    elif echo "$file_info" | grep -q 'stripped'; then
        log_warn "Binary is stripped — function names will be missing"
    fi

    echo ""
}

# ── parse args ──────────────────────────────────────────────────────────────
PROGRAM=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage ;;
        --tool)
            TOOL="$2"; shift 2 ;;
        --mode)
            MODE="$2"; shift 2 ;;
        --out)
            OUT_DIR="$2"; shift 2 ;;
        --)
            shift; PROGRAM=("$@"); break ;;
        *)
            PROGRAM+=("$1"); shift ;;
    esac
done

if [ ${#PROGRAM[@]} -eq 0 ]; then
    log_err "No program specified. Use -- <program> [args...]"
    usage
    exit 1
fi

# ── auto-detect tool ────────────────────────────────────────────────────────
if [ "$TOOL" = "auto" ]; then
    TOOL=$(detect_tool)
    log_info "Auto-detected tool: ${TOOL}"
fi

# ── tool-specific setup & execution ─────────────────────────────────────────
run_vtune() {
    source /opt/intel/oneapi/vtune/latest/vtune-vars.sh

    local mode_map
    case "${MODE}" in
        default|hotspots)        mode_map="hotspots" ;;
        uarch)                   mode_map="uarch-exploration" ;;
        memory-access)           mode_map="memory-access" ;;
        memory-consumption)      mode_map="memory-consumption" ;;
        threading)               mode_map="threading" ;;
        hpc-performance|hpc)     mode_map="hpc-performance" ;;
        io)                      mode_map="io" ;;
        *) log_err "Unknown VTune mode: ${MODE}"; exit 1 ;;
    esac

    if [ -z "${OUT_DIR}" ]; then
        OUT_DIR="./vtune_${mode_map}_$(date +%Y%m%d_%H%M%S)"
    fi

    log_info "VTune mode: ${mode_map}"
    log_info "Output dir: ${OUT_DIR}"
    log_cmd "vtune -collect ${mode_map} -result-dir ${OUT_DIR} -- ${PROGRAM[*]}"
    echo ""

    # Disable 'set -e' for the profiling run so we can catch interruption
    # and still print results path
    trap on_interrupt INT TERM
    set +e
    vtune -collect "${mode_map}" -result-dir "${OUT_DIR}" -- "${PROGRAM[@]}"
    local rc=$?
    set -e
    trap - INT TERM

    echo ""
    if [ $rc -eq 0 ] || [ $INTERRUPTED -eq 1 ] || [ $rc -eq 4 ]; then
        if [ -d "${OUT_DIR}" ]; then
            local reason=""
            if [ $INTERRUPTED -eq 1 ]; then reason=" (interrupted by user)"
            elif [ $rc -eq 4 ]; then reason=" (program terminated, collection stopped)"
            fi
            log_info "Results saved to: ${OUT_DIR}${reason}"
            log_info "Open with: vtune-gui ${OUT_DIR}"
        fi
    else
        log_err "VTune exited with code ${rc}"
    fi
}

run_perf() {
    trap on_interrupt INT TERM

    case "${MODE}" in
        default|record)
            local out_file="${OUT_DIR:-perf.data}"
            log_info "Perf mode: record (press Ctrl+C to stop)"
            log_info "Output: ${out_file}"
            log_cmd "perf record -g -o ${out_file} -- ${PROGRAM[*]}"
            echo ""

            set +e
            perf record -g -o "${out_file}" -- "${PROGRAM[@]}"
            local rc=$?
            set -e

            echo ""
            if [ $rc -eq 0 ] || [ $INTERRUPTED -eq 1 ]; then
                if [ -f "${out_file}" ]; then
                    log_info "Results saved to: ${out_file}"
                    log_info "View with: perf report -i ${out_file}"
                fi
            else
                log_err "perf exited with code ${rc}"
            fi
            ;;
        stat)
            log_info "Perf mode: stat"
            log_cmd "perf stat -- ${PROGRAM[*]}"
            echo ""

            set +e
            perf stat -- "${PROGRAM[@]}"
            local rc=$?
            set -e

            if [ $rc -ne 0 ] && [ $INTERRUPTED -ne 1 ]; then
                log_err "perf stat exited with code ${rc}"
            fi
            ;;
        mem)
            local out_file="${OUT_DIR:-perf_mem.data}"
            log_info "Perf mode: mem (cache + instructions, press Ctrl+C to stop)"
            log_info "Output: ${out_file}"
            log_cmd "perf record -e cpu-cycles,instructions,cache-references,cache-misses -g -o ${out_file} -- ${PROGRAM[*]}"
            echo ""

            set +e
            perf record -e cpu-cycles,instructions,cache-references,cache-misses -g -o "${out_file}" -- "${PROGRAM[@]}"
            local rc=$?
            set -e

            echo ""
            if [ $rc -eq 0 ] || [ $INTERRUPTED -eq 1 ]; then
                if [ -f "${out_file}" ]; then
                    log_info "Results saved to: ${out_file}"
                    log_info "View with: perf report -i ${out_file}"
                fi
            else
                log_err "perf exited with code ${rc}"
            fi
            ;;
        annotate)
            local src="${OUT_DIR:-perf.data}"
            if [ ! -f "${src}" ]; then
                log_err "perf.data not found. Run 'perf record' first."
                exit 1
            fi
            log_cmd "perf annotate -i ${src}"
            perf annotate -i "${src}"
            ;;
        *)
            log_err "Unknown perf mode: ${MODE}"; exit 1 ;;
    esac

    trap - INT TERM
}

# ── kernel settings check ─────────────────────────────────────────────────
check_kernel() {
    local tool="$1"
    local ptrace_scope
    ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo "unknown")
    local perf_paranoid
    perf_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")

    if [ "$tool" = "vtune" ] && [ "$ptrace_scope" != "0" ]; then
        log_warn "ptrace_scope = ${ptrace_scope}, setting to 0 for VTune..."
        sudo sysctl -w kernel.yama.ptrace_scope=0
    fi

    if [ "$tool" = "perf" ] && [ "$perf_paranoid" -gt 2 ] 2>/dev/null; then
        log_warn "perf_event_paranoid = ${perf_paranoid}, setting to 2 for perf..."
        sudo sysctl -w kernel.perf_event_paranoid=2
    fi
}

# ── run ─────────────────────────────────────────────────────────────────────
check_binary "${PROGRAM[0]}"
check_kernel "${TOOL}"
echo ""

case "${TOOL}" in
    vtune) run_vtune ;;
    perf)  run_perf ;;
    *)
        log_err "Unknown tool: ${TOOL}. Choose vtune or perf."
        exit 1
        ;;
esac
