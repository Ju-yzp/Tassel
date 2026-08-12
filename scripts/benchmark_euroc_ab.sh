#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
    echo "Usage: $0 BASELINE_BINARY CANDIDATE_BINARY CONFIG DATASET [RUNS] [HZ] [OUTPUT_DIR]" >&2
    exit 2
fi

baseline=$(realpath "$1")
candidate=$(realpath "$2")
config=$(realpath "$3")
dataset=$(realpath "$4")
runs=${5:-3}
hz=${6:-60}
output_dir=${7:-/tmp/tassel-euroc-ab-$(date +%Y%m%d-%H%M%S)}

for binary in "$baseline" "$candidate"; do
    if [[ ! -x "$binary" ]]; then
        echo "Benchmark binary is not executable: $binary" >&2
        exit 2
    fi
done
if [[ ! -f "$config" || ! -d "$dataset" ]]; then
    echo "Config or dataset path is invalid" >&2
    exit 2
fi
if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$hz" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$hz" == "0" ]]; then
    echo "HZ must be positive" >&2
    exit 2
fi

mkdir -p "$output_dir"
results="$output_dir/results.tsv"
printf 'variant\trun\tceres_calls\tresidual_ms\tjacobian_ms\tlinear_solver_ms\tpreprocessor_ms\tceres_total_ms\testimator_calls\testimator_ms\tstage_samples\tfeature_ms\tpredict_ms\ttriangulation_ms\tstage_optimization_ms\toutlier_ms\tmarginalization_ms\tmigration_ms\tnon_optimization_ms\tstage_total_ms\titerations\tsuccessful\trejected\treduced_residuals\treduced_parameter_blocks\tate_rmse_m\tterminal_error_m\trotation_rmse_rad\n' >"$results"

summarize_log() {
    local variant=$1
    local run=$2
    local log=$3
    awk -v variant="$variant" -v run="$run" '
        function value(prefix,    i, item) {
            for (i = 1; i <= NF; ++i) {
                item = $i
                if (index(item, prefix) == 1) {
                    sub(prefix, "", item)
                    sub(/ms$/, "", item)
                    return item + 0
                }
            }
            return 0
        }
        function reduced_value(prefix,    i, item) {
            for (i = 1; i <= NF; ++i) {
                item = $i
                if (index(item, prefix) == 1) {
                    sub(prefix, "", item)
                    sub(/^[^/]*\//, "", item)
                    return item + 0
                }
            }
            return 0
        }
        /Ceres summary:/ {
            ++ceres_count
            residual += value("residual=")
            jacobian += value("jacobian=")
            linear_solver += value("linear_solver=")
            preprocessor += value("preprocessor=")
            ceres_total += value("total=")
            iterations += value("iterations=")
            successful += value("successful=")
            rejected += value("rejected=")
            reduced_residuals += reduced_value("residuals=")
            reduced_parameter_blocks += reduced_value("parameter_blocks=")
        }
        /Timing pipeline:/ {
            ++estimator_count
            estimator += value("estimator=")
        }
        /Estimator stages:/ {
            samples = value("samples=")
            stage_samples += samples
            feature += samples * value("feature=")
            predict += samples * value("predict=")
            triangulation += samples * value("triangulation=")
            stage_optimization += samples * value("optimization=")
            outlier += samples * value("outlier=")
            marginalization += samples * value("marginalization=")
            migration += samples * value("migration=")
            non_optimization += samples * value("non_optimization=")
            stage_total += samples * value("total=")
        }
        /trajectory evaluation/ {
            ate = value("ATE_RMSE=")
            terminal = value("terminal_position_error=")
            rotation = value("rotation_RMSE=")
        }
        END {
            if (ceres_count == 0 || estimator_count == 0 || stage_samples == 0 || ate == 0) {
                exit 3
            }
            printf "%s\t%d\t%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%d\t%.9f\t%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\n",
                variant, run, ceres_count, residual / ceres_count, jacobian / ceres_count,
                linear_solver / ceres_count, preprocessor / ceres_count,
                ceres_total / ceres_count, estimator_count, estimator / estimator_count,
                stage_samples, feature / stage_samples, predict / stage_samples,
                triangulation / stage_samples, stage_optimization / stage_samples,
                outlier / stage_samples, marginalization / stage_samples,
                migration / stage_samples, non_optimization / stage_samples,
                stage_total / stage_samples, iterations / ceres_count,
                successful / ceres_count, rejected / ceres_count,
                reduced_residuals / ceres_count, reduced_parameter_blocks / ceres_count,
                ate, terminal, rotation
        }
    ' "$log" >>"$results"
}

run_one() {
    local variant=$1
    local run=$2
    local binary=$3
    local log="$output_dir/${variant}_${run}.log"
    echo "[$variant $run/$runs] $binary"
    "$binary" "$config" "$dataset" "$hz" >"$log" 2>&1
    summarize_log "$variant" "$run" "$log"
}

validate_pair() {
    local run=$1
    local baseline_counts candidate_counts
    baseline_counts=$(awk -F '\t' -v run="$run" '$1 == "baseline" && $2 == run { print $3, $9 }' "$results")
    candidate_counts=$(awk -F '\t' -v run="$run" '$1 == "candidate" && $2 == run { print $3, $9 }' "$results")
    if [[ "$baseline_counts" != "$candidate_counts" ]]; then
        echo "Run $run sample-count mismatch: baseline=[$baseline_counts] candidate=[$candidate_counts]" >&2
        exit 3
    fi
}

for ((run = 1; run <= runs; ++run)); do
    if ((run % 2 == 1)); then
        run_one baseline "$run" "$baseline"
        run_one candidate "$run" "$candidate"
    else
        run_one candidate "$run" "$candidate"
        run_one baseline "$run" "$baseline"
    fi
    validate_pair "$run"
done

awk -F '\t' '
    NR == 1 { next }
    {
        variant = $1
        count[variant]++
        for (column = 4; column <= 28; ++column) {
            key = variant SUBSEP column
            sum[key] += $column
            square_sum[key] += $column * $column
        }
        ceres_calls[variant] += $3
        estimator_calls[variant] += $9
        stage_samples[variant] += $11
        ceres_total[$1, $2] = $8
    }
    function mean(variant, column) {
        return sum[variant SUBSEP column] / count[variant]
    }
    function sample_sd(variant, column,    n, average, variance) {
        n = count[variant]
        if (n < 2) {
            return 0
        }
        average = mean(variant, column)
        variance = (square_sum[variant SUBSEP column] - n * average * average) / (n - 1)
        return variance > 0 ? sqrt(variance) : 0
    }
    function change(column) {
        return 100 * (mean("candidate", column) / mean("baseline", column) - 1)
    }
    function metric_unit(column) {
        if (column == 26 || column == 27) {
            return "m"
        }
        if (column == 28) {
            return "rad"
        }
        if ((column >= 4 && column <= 8) || column == 10 ||
            (column >= 12 && column <= 20)) {
            return "ms"
        }
        return ""
    }
    END {
        columns[1] = 4; names[1] = "Residual"
        columns[2] = 5; names[2] = "Jacobian"
        columns[3] = 6; names[3] = "Linear solver"
        columns[4] = 7; names[4] = "Preprocessor"
        columns[5] = 8; names[5] = "Ceres total"
        columns[6] = 10; names[6] = "Estimator"
        columns[7] = 12; names[7] = "Feature management"
        columns[8] = 13; names[8] = "IMU prediction"
        columns[9] = 14; names[9] = "Triangulation"
        columns[10] = 15; names[10] = "Stage optimization"
        columns[11] = 16; names[11] = "Outlier rejection"
        columns[12] = 17; names[12] = "Marginalization prior"
        columns[13] = 18; names[13] = "Window migration"
        columns[14] = 19; names[14] = "Non-optimization total"
        columns[15] = 20; names[15] = "Stage total"
        columns[16] = 21; names[16] = "Iterations / solve"
        columns[17] = 22; names[17] = "Successful / solve"
        columns[18] = 23; names[18] = "Rejected / solve"
        columns[19] = 24; names[19] = "Reduced residuals"
        columns[20] = 25; names[20] = "Reduced parameter blocks"
        columns[21] = 26; names[21] = "ATE RMSE"
        columns[22] = 27; names[22] = "Terminal error"
        columns[23] = 28; names[23] = "Rotation RMSE"
        print "| Metric | Baseline mean +/- SD | Candidate mean +/- SD | Change |"
        print "|---|---:|---:|---:|"
        for (i = 1; i <= 23; ++i) {
            column = columns[i]
            unit = metric_unit(column)
            printf "| %s | %.3f +/- %.3f %s | %.3f +/- %.3f %s | %+.2f%% |\n",
                names[i], mean("baseline", column), sample_sd("baseline", column), unit,
                mean("candidate", column), sample_sd("candidate", column), unit, change(column)
        }
        print ""
        print "Paired Ceres-total changes:"
        for (run = 1; run <= count["baseline"]; ++run) {
            printf "run%d %+.2f%%\n", run,
                100 * (ceres_total["candidate", run] / ceres_total["baseline", run] - 1)
        }
        printf "\nCeres calls: baseline=%d candidate=%d\n",
            ceres_calls["baseline"], ceres_calls["candidate"]
        printf "Estimator samples: baseline=%d candidate=%d\n",
            estimator_calls["baseline"], estimator_calls["candidate"]
        printf "Estimator stage samples: baseline=%d candidate=%d\n",
            stage_samples["baseline"], stage_samples["candidate"]
    }
' "$results" | tee "$output_dir/summary.md"

echo "Logs and summary: $output_dir"
