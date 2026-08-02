#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
V="${V:-1024}"
S="${S:-32}"
Q="${Q:-4}"
UNIT_SIZE="${UNIT_SIZE:-1000}"
TRAIN_UNITS="${1:-${TRAIN_UNITS:-1000}}"
TEST_UNITS="${2:-${TEST_UNITS:-20}}"
BASE_SEED="${BASE_SEED:-20260715}"
OUTPUT_DIR="${3:-${OUTPUT_DIR:-phase_c_random_data/V${V}_S${S}_q${Q}_seed${BASE_SEED}}}"

echo "Generating random unit dataset"
echo "  V=${V}"
echo "  S=${S}"
echo "  q=${Q}"
echo "  unit_size=${UNIT_SIZE}"
echo "  train_units=${TRAIN_UNITS}"
echo "  test_units=${TEST_UNITS}"
echo "  base_seed=${BASE_SEED}"
echo "  output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" -m phase_c.data.cli random-units \
  --V "${V}" \
  --S "${S}" \
  --q "${Q}" \
  --unit-size "${UNIT_SIZE}" \
  --train-units "${TRAIN_UNITS}" \
  --test-units "${TEST_UNITS}" \
  --base-seed "${BASE_SEED}" \
  --output-dir "${OUTPUT_DIR}"
