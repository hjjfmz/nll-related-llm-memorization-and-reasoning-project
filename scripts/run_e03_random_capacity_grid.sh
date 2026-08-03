#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DATASET_ROOT="${DATASET_ROOT:-phase_c_random_data/V2048_S64_seed20260715}"
OUTPUT_ROOT="${OUTPUT_ROOT:-phase_c_runs/e03_paper_grid}"
TEST_UNITS="${TEST_UNITS:-5}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-32}"
MAX_STEPS="${MAX_STEPS:-250000}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MINIMUM_LEARNING_RATE="${MINIMUM_LEARNING_RATE:-${LEARNING_RATE}}"
MODEL_SEED="${MODEL_SEED:-20270619}"
SAMPLING_SEED="${SAMPLING_SEED:-20270618}"
CURRENT_PID=""

H_BITS_PER_SAMPLE=704
UNIT_SIZE=1000
RUN_SPECS=(
  "L2_H128:1:725632"
  "L2_H128:3:725632"
  "L2_H128:5:725632"
  "L2_H128:10:725632"
  "L4_H128:1:1122176"
  "L4_H128:4:1122176"
  "L4_H128:7:1122176"
  "L4_H128:13:1122176"
  "L8_H256:5:6976256"
  "L8_H256:20:6976256"
  "L8_H256:40:6976256"
  "L8_H256:80:6976256"
)

cleanup() {
  if [[ -n "${CURRENT_PID}" ]] && kill -0 "${CURRENT_PID}" 2>/dev/null; then
    echo "Interrupt received; stopping active torchrun pid=${CURRENT_PID}."
    kill -TERM "${CURRENT_PID}" 2>/dev/null || true
    wait "${CURRENT_PID}" 2>/dev/null || true
  fi
  echo "Active training process exited; its CUDA context is released."
  exit 130
}
trap cleanup INT TERM

GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION * NPROC_PER_NODE))

if [[ ! -f "${DATASET_ROOT}/dataset_manifest.json" ]]; then
  echo "Missing dataset manifest: ${DATASET_ROOT}/dataset_manifest.json" >&2
  echo "Generate data first: bash generate_random_units.sh 100 ${TEST_UNITS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "E03 random-capacity grid"
echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "  processes=${NPROC_PER_NODE}"
echo "  dataset_root=${DATASET_ROOT}"
echo "  output_root=${OUTPUT_ROOT}"
echo "  global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "  max_steps=${MAX_STEPS}"

for run_spec in "${RUN_SPECS[@]}"; do
  IFS=":" read -r model train_units total_parameters <<< "${run_spec}"
  train_samples=$((train_units * UNIT_SIZE))
  run_dir="${OUTPUT_ROOT}/${model}_N${train_samples}_steps${MAX_STEPS}"
  entropy_ratio=$(awk -v n="${train_samples}" -v h="${H_BITS_PER_SAMPLE}" -v p="${total_parameters}" 'BEGIN { printf "%.3f", n * h / p }')

  if [[ ! -f "${DATASET_ROOT}/train/${train_units}.jsonl.gz" ]]; then
    echo "Missing train unit ${train_units}: ${DATASET_ROOT}/train/${train_units}.jsonl.gz" >&2
    exit 1
  fi
  if [[ ! -f "${DATASET_ROOT}/test/${TEST_UNITS}.jsonl.gz" ]]; then
    echo "Missing test unit ${TEST_UNITS}: ${DATASET_ROOT}/test/${TEST_UNITS}.jsonl.gz" >&2
    exit 1
  fi
  if [[ -f "${run_dir}/final_metrics.json" ]]; then
    echo "Skipping completed run: ${run_dir}"
    continue
  fi

  echo
  echo "============================================================"
  echo "Starting ${model}, N=${train_samples}, H_over_P=${entropy_ratio}"
  echo "  micro_batch=${MICRO_BATCH_SIZE}, accumulation=${GRADIENT_ACCUMULATION}, global_batch=${GLOBAL_BATCH_SIZE}"
  echo "  max_steps=${MAX_STEPS}, model_seed=${MODEL_SEED}, sampling_seed=${SAMPLING_SEED}"
  echo "  output_dir=${run_dir}"

  if [[ -f "${run_dir}/checkpoint_latest.pt" && -f "${run_dir}/run_config.json" ]]; then
    command=(
      "${PYTHON_BIN}" -m torch.distributed.run --standalone
      "--nproc_per_node=${NPROC_PER_NODE}" -m phase_c.cli random resume
      --run-dir "${run_dir}"
    )
  else
    command=(
      "${PYTHON_BIN}" -m torch.distributed.run --standalone
      "--nproc_per_node=${NPROC_PER_NODE}" -m phase_c.cli random train
      --model "${model}"
      --dataset-root "${DATASET_ROOT}"
      --train-units "${train_units}"
      --test-units "${TEST_UNITS}"
      --micro-batch-size "${MICRO_BATCH_SIZE}"
      --gradient-accumulation "${GRADIENT_ACCUMULATION}"
      --max-steps "${MAX_STEPS}"
      --learning-rate "${LEARNING_RATE}"
      --minimum-learning-rate "${MINIMUM_LEARNING_RATE}"
      --model-seed "${MODEL_SEED}"
      --sampling-seed "${SAMPLING_SEED}"
      --eval-size 0
      --output-dir "${run_dir}"
    )
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${command[@]}" &
  CURRENT_PID=$!
  wait "${CURRENT_PID}"
  CURRENT_PID=""
  echo "Completed ${model}, N=${train_samples}; torchrun exited and released GPU memory."
done

echo "All requested E03 runs are complete."
