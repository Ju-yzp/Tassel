#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/archive_benchmark_binary.sh --build-dir BUILD_DIR --target RELATIVE_BINARY --tag TAG --note NOTE [--out OUT_DIR]

Example:
  scripts/archive_benchmark_binary.sh \
    --build-dir build-vtune \
    --target tassel_core/test_euroc \
    --tag schur-plan-baseline \
    --note "Before Schur layout-plan cache"
USAGE
}

BUILD_DIR=""
TARGET=""
TAG=""
NOTE=""
OUT_ROOT="artifacts/binaries"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --note)
            NOTE="$2"
            shift 2
            ;;
        --out)
            OUT_ROOT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$BUILD_DIR" || -z "$TARGET" || -z "$TAG" || -z "$NOTE" ]]; then
    usage >&2
    exit 2
fi

if [[ ! "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "TAG must contain only letters, numbers, '.', '_' or '-'" >&2
    exit 2
fi

binary="${BUILD_DIR%/}/${TARGET}"
if [[ ! -x "$binary" ]]; then
    echo "Target binary is not executable: $binary" >&2
    exit 2
fi

timestamp=$(date +%Y%m%d-%H%M%S)
tassel_commit=$(git rev-parse --short=12 HEAD)
ceres_commit="none"
if [[ -d third_party/ceres-solver ]]; then
    ceres_commit=$(git -C third_party/ceres-solver rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")
fi

artifact_name="$(basename "$TARGET")__${TAG}__tassel-${tassel_commit}__ceres-${ceres_commit}__${timestamp}"
artifact_dir="${OUT_ROOT%/}/${artifact_name}"
mkdir -p "$artifact_dir/bin" "$artifact_dir/lib" "$artifact_dir/meta"

cp -L "$binary" "$artifact_dir/bin/"

for lib in \
    "${BUILD_DIR%/}/tassel_core/libtassel_core.so" \
    "${BUILD_DIR%/}/tassel_tools/libtassel_tools.so" \
    "${BUILD_DIR%/}/tassel_hardware/libtassel_hardware.so"; do
    if [[ -f "$lib" ]]; then
        cp -L "$lib" "$artifact_dir/lib/"
    fi
done

cat >"$artifact_dir/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
DIR=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH="\${DIR}/lib:\${LD_LIBRARY_PATH:-}"
exec "\${DIR}/bin/$(basename "$TARGET")" "\$@"
EOF
chmod +x "$artifact_dir/run.sh"

sha256sum "$artifact_dir/bin/$(basename "$TARGET")" >"$artifact_dir/meta/sha256.txt"
if compgen -G "$artifact_dir/lib/*.so" >/dev/null; then
    sha256sum "$artifact_dir"/lib/*.so >>"$artifact_dir/meta/sha256.txt"
fi

{
    echo "# ${artifact_name}"
    echo
    echo "- tag: \`${TAG}\`"
    echo "- note: ${NOTE}"
    echo "- timestamp: \`${timestamp}\`"
    echo "- tassel_commit: \`${tassel_commit}\`"
    echo "- ceres_commit: \`${ceres_commit}\`"
    echo "- build_dir: \`${BUILD_DIR}\`"
    echo "- target: \`${TARGET}\`"
    echo "- executable: \`bin/$(basename "$TARGET")\`"
    echo "- wrapper: \`run.sh\`"
    echo
    echo "## Git Status"
    echo
    echo '```text'
    git status --short
    echo '```'
    echo
    echo "## Submodule Status"
    echo
    echo '```text'
    git submodule status 2>/dev/null || true
    echo '```'
    echo
    echo "## File Info"
    echo
    echo '```text'
    file "$binary"
    echo '```'
    echo
    echo "## Dynamic Links"
    echo
    echo '```text'
    ldd "$binary"
    echo '```'
    echo
    echo "## CMake Cache Summary"
    echo
    echo '```text'
    cmake -LA -N "$BUILD_DIR" | grep -E '^(CMAKE_BUILD_TYPE|TASSEL_|BUILD_TESTING)' || true
    echo '```'
} >"$artifact_dir/README.md"

cmake -LA -N "$BUILD_DIR" >"$artifact_dir/meta/cmake-cache.txt"
readelf -d "$binary" >"$artifact_dir/meta/readelf-dynamic.txt"
ldd "$binary" >"$artifact_dir/meta/ldd.txt"
git diff --stat >"$artifact_dir/meta/git-diff-stat.txt"
git diff >"$artifact_dir/meta/git-diff.patch"

echo "$artifact_dir"
