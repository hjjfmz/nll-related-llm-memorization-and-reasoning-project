#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
V="${V:-2048}"
L="${L:-4}"
D="${D:-2}"
UNIT_SIZE="${UNIT_SIZE:-1000}"
TRAIN_UNITS="${1:-${TRAIN_UNITS:-100}}"
VALIDATION_UNITS="${2:-${VALIDATION_UNITS:-10}}"
TEST_UNITS="${3:-${TEST_UNITS:-20}}"
BASE_SEED="${BASE_SEED:-20260715}"
OUTPUT_DIR="${4:-${OUTPUT_DIR:-phase_c_dag_data/V${V}_L${L}_d${D}_seed${BASE_SEED}}}"

echo "Generating DAG unit dataset"
echo "  V=${V}"
echo "  L=${L}"
echo "  d=${D}"
echo "  unit_size=${UNIT_SIZE}"
echo "  train_units=${TRAIN_UNITS}"
echo "  validation_units=${VALIDATION_UNITS}"
echo "  test_units=${TEST_UNITS}"
echo "  base_seed=${BASE_SEED}"
echo "  output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" -m phase_c.cli dag gen-data \
  --V "${V}" \
  --L "${L}" \
  --d "${D}" \
  --unit-size "${UNIT_SIZE}" \
  --train-units "${TRAIN_UNITS}" \
  --validation-units "${VALIDATION_UNITS}" \
  --test-units "${TEST_UNITS}" \
  --base-seed "${BASE_SEED}" \
  --output-dir "${OUTPUT_DIR}"
