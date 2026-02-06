#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REF_DIR="$ROOT_DIR/.ref"
BUILD_DIR="$REF_DIR/build"
BIN_DIR="$REF_DIR/bin"

find_ref_src() {
    if [ "${BITNET_REF_SRC:-}" != "" ]; then
        printf '%s\n' "$BITNET_REF_SRC"
        return 0
    fi

    for candidate in \
        "$ROOT_DIR" \
        "$ROOT_DIR/upstream/BitNet" \
        "$ROOT_DIR/.ref/bitnet.cpp" \
        "$ROOT_DIR/../BitNet" \
        "$ROOT_DIR/../bitnet.cpp"
    do
        if [ -f "$candidate/CMakeLists.txt" ] && [ -d "$candidate/src" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required but not installed" >&2
    exit 1
fi

REF_SRC=$(find_ref_src || true)
if [ "$REF_SRC" = "" ]; then
    cat >&2 <<'EOM'
Could not locate upstream BitNet C++ source tree.
Set BITNET_REF_SRC to a checkout path that contains CMakeLists.txt and src/.
EOM
    exit 1
fi

if [ -d "$REF_SRC/3rdparty/llama.cpp" ] && [ ! -f "$REF_SRC/3rdparty/llama.cpp/CMakeLists.txt" ]; then
    cat >&2 <<EOM
Reference source is missing required submodule contents:
  $REF_SRC/3rdparty/llama.cpp

Run:
  git -C "$REF_SRC" submodule update --init --recursive
EOM
    exit 1
fi

KERNEL_HEADER="$REF_SRC/include/bitnet-lut-kernels.h"
if [ ! -f "$KERNEL_HEADER" ]; then
    if [ "${BITNET_REF_AUTOGEN_KERNEL_HEADER:-0}" = "1" ]; then
        PRESET_KERNEL=${BITNET_REF_PRESET_KERNEL:-}
        if [ "$PRESET_KERNEL" = "" ]; then
            case "$(uname -m)" in
                x86_64|amd64)
                    PRESET_KERNEL="$REF_SRC/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl2.h"
                    ;;
                aarch64|arm64)
                    PRESET_KERNEL="$REF_SRC/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h"
                    ;;
            esac
        fi
        if [ "$PRESET_KERNEL" != "" ] && [ -f "$PRESET_KERNEL" ]; then
            cp "$PRESET_KERNEL" "$KERNEL_HEADER"
            echo "Generated $KERNEL_HEADER from preset: $PRESET_KERNEL"
        else
            echo "Missing preset kernel header. Set BITNET_REF_PRESET_KERNEL to a valid preset file." >&2
            exit 1
        fi
    else
        cat >&2 <<EOM
Missing generated kernel header:
  $KERNEL_HEADER

Run upstream setup first (recommended):
  python "$REF_SRC/setup_env.py" -md <model_dir> -q i2_s

Or allow this script to copy a preset header:
  BITNET_REF_AUTOGEN_KERNEL_HEADER=1 ./scripts/build_ref.sh
EOM
        exit 1
    fi
fi

mkdir -p "$BUILD_DIR" "$BIN_DIR"

BUILD_TYPE=${BITNET_REF_BUILD_TYPE:-Release}
TARGET=${BITNET_REF_TARGET:-llama-cli}

cmake -S "$REF_SRC" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
if [ "$TARGET" != "" ]; then
    cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --target "$TARGET"
else
    cmake --build "$BUILD_DIR" --config "$BUILD_TYPE"
fi

REF_BIN=${BITNET_REF_BIN:-}
if [ "$REF_BIN" = "" ]; then
    for name in llama-cli bitnet main; do
        found=$(find "$BUILD_DIR" -type f -name "$name" 2>/dev/null | head -n 1 || true)
        if [ "$found" != "" ] && [ -x "$found" ]; then
            REF_BIN=$found
            break
        fi
    done
fi

if [ "$REF_BIN" = "" ]; then
    found=$(find "$BUILD_DIR" -type f -perm -111 2>/dev/null | grep -v '\.so$' | grep -v '\.a$' | head -n 1 || true)
    if [ "$found" != "" ]; then
        REF_BIN=$found
    fi
fi

if [ "$REF_BIN" = "" ] || [ ! -x "$REF_BIN" ]; then
    echo "Build succeeded but failed to locate a runnable reference binary." >&2
    echo "Set BITNET_REF_BIN to the executable path and rerun." >&2
    exit 1
fi

cp "$REF_BIN" "$BIN_DIR/ref-infer"

cat > "$REF_DIR/ref.env" <<EOM
ROOT_DIR=$ROOT_DIR
REF_SRC=$REF_SRC
REF_BUILD_DIR=$BUILD_DIR
REF_BIN=$BIN_DIR/ref-infer
EOM

echo "Reference binary ready: $BIN_DIR/ref-infer"
