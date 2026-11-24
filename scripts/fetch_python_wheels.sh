#!/usr/bin/env bash
# Fetch Python wheels on a networked machine, then install them offline on LumenX.
#
# Recommended: run this on a machine with the SAME OS/arch/Python as the target
# (this project targets macOS arm64 + Python 3.15) using --native.
#
#   ./fetch_python_wheels.sh --native
#
# If the download machine differs (e.g. Linux x86_64), pass a cross target:
#
#   ./fetch_python_wheels.sh \
#     --platform macosx_14_0_arm64 --python-version 3.15 --implementation cp --abi cp315
#
# Then copy the wheel folder to the LumenX machine and install offline:
#
#   python3 -m venv .venv-local
#   .venv-local/bin/pip install --no-index --find-links ./lumenx_wheels \
#       -r requirements-local.txt requests
set -euo pipefail

OUT="./lumenx_wheels"
MIRROR="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
REQS=("requirements-local.txt")
EXTRA=("requests")
NATIVE=0
PLATFORM=""
PYVER=""
IMPL="cp"
ABI=""
BENCHMARK=0
MIRROR_SET=0

bench_mirrors() {
  local mirrors=(
    "https://mirrors.aliyun.com/pypi/simple"
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    "https://pypi.mirrors.ustc.edu.cn/simple"
    "https://mirrors.cloud.tencent.com/pypi/simple"
    "https://repo.huaweicloud.com/repository/pypi/simple"
    "https://pypi.org/simple"
  )
  local best="" best_speed=0
  for base in "${mirrors[@]}"; do
    local href abs speed
    href=$(curl -sL --max-time 12 "$base/babel/" | grep -oE 'href="[^"]*Babel-[0-9.]+-py3-none-any\.whl[^"]*"' | tail -1 | sed 's/href="//;s/"$//')
    if [[ -z "$href" ]]; then
      echo "  $(printf '%-55s' "$base") unreachable" >&2
      continue
    fi
    abs=$(python3 - "$base/babel/" "$href" <<'PY'
import sys, urllib.parse
print(urllib.parse.urljoin(sys.argv[1], sys.argv[2]))
PY
)
    speed=$(curl -sL --max-time 8 -o /dev/null -w "%{speed_download}" "$abs")
    echo "  $(printf '%-55s' "$base") $(awk -v s="$speed" 'BEGIN{printf "%.2f MB/s", s/1048576}')" >&2
    if awk -v s="$speed" -v b="$best_speed" 'BEGIN{exit !(s>b)}'; then
      best_speed=$speed
      best=$base
    fi
  done
  echo "$best"
}

usage() {
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --native                      download for the current interpreter/platform
  --platform PLATFORM           e.g. macosx_14_0_arm64 (with --only-binary)
  --python-version VER          e.g. 3.15
  --implementation IMPL         e.g. cp
  --abi ABI                     e.g. cp315
  --out DIR                     output wheel dir (default ./lumenx_wheels)
  --mirror URL                  pip index url
  --benchmark                   probe mirrors and use the fastest
  --requirements LIST           comma-separated requirement files
  --extra LIST                  extra package names (default: requests)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --native) NATIVE=1; shift ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --python-version) PYVER="$2"; shift 2 ;;
    --implementation) IMPL="$2"; shift 2 ;;
    --abi) ABI="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --mirror) MIRROR="$2"; MIRROR_SET=1; shift 2 ;;
    --benchmark) BENCHMARK=1; shift ;;
    --requirements) IFS=',' read -ra REQS <<< "$2"; shift 2 ;;
    --extra) IFS=',' read -ra EXTRA <<< "$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$BENCHMARK" -eq 1 && "$MIRROR_SET" -eq 0 ]]; then
  echo "benchmarking mirrors (8s probe each)..."
  BEST=$(bench_mirrors)
  if [[ -n "$BEST" ]]; then
    MIRROR="$BEST"
    echo "fastest mirror: $MIRROR"
  else
    echo "benchmark failed; keeping default $MIRROR"
  fi
fi

PYTHON="${PYTHON:-python3}"
mkdir -p "$OUT"

echo "python : $($PYTHON --version 2>&1)"
echo "mirror : $MIRROR"
echo "out    : $OUT"
echo "reqs   : ${REQS[*]}"
echo "extra  : ${EXTRA[*]}"

ARGS=(-d "$OUT" -i "$MIRROR")
if [[ "$NATIVE" -eq 1 ]]; then
  echo "mode   : native"
else
  if [[ -z "$PLATFORM" || -z "$PYVER" || -z "$ABI" ]]; then
    echo "ERROR: cross-download needs --platform, --python-version and --abi" >&2
    echo "       (or use --native on a matching machine)" >&2
    exit 2
  fi
  echo "mode   : cross -> platform=$PLATFORM python=$PYVER impl=$IMPL abi=$ABI"
  ARGS+=(--only-binary=:all: --platform "$PLATFORM" --python-version "$PYVER" --implementation "$IMPL" --abi "$ABI")
fi

for req in "${REQS[@]}"; do
  if [[ ! -f "$req" ]]; then
    echo "ERROR: requirement file not found: $req" >&2
    exit 2
  fi
  ARGS+=(-r "$req")
done
ARGS+=("${EXTRA[@]}")

echo "running: $PYTHON -m pip download ${ARGS[*]}"
"$PYTHON" -m pip download "${ARGS[@]}"

echo
echo "wheels:"
ls -1 "$OUT" | sed 's/^/  /' | head -60
count=$(find "$OUT" -maxdepth 1 -type f | wc -l | tr -d ' ')
size=$(du -sh "$OUT" | cut -f1)
echo "total: $count files, $size"
echo
echo "Next on the LumenX machine:"
echo "  rsync -av $OUT/ /path/on/lumenx/lumenx_wheels/"
echo "  python3 -m venv .venv-local"
echo "  .venv-local/bin/pip install --no-index --find-links ./lumenx_wheels -r requirements-local.txt requests"
