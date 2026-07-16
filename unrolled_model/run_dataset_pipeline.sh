#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash unrolled_model/run_dataset_pipeline.sh
#   MODE=generate bash unrolled_model/run_dataset_pipeline.sh
#   MODE=inspect DATA_PATH=./train/my_dataset.pt bash unrolled_model/run_dataset_pipeline.sh
#   NUM_NODES=100 N_DATA=50 K0=10 K1=4 DATASET_SIZE=200 bash unrolled_model/run_dataset_pipeline.sh
#
# Future training hook:
#   MODE=train TRAIN_SCRIPT=unrolled_model/train.py DATA_PATH=... bash unrolled_model/run_dataset_pipeline.sh

MODE="${MODE:-all}"  # generate | inspect | all | train

SET_TYPE="${SET_TYPE:-train}"
NUM_NODES="${NUM_NODES:-100}"
N_DATA="${N_DATA:-50}"
K0="${K0:-10}"
K1="${K1:-4}"
DATASET_SIZE="${DATASET_SIZE:-100}"
RANDOM_STATE="${RANDOM_STATE:-0}"
SAME_GRAPH="${SAME_GRAPH:-0}"
SAMPLE_MODE="${SAMPLE_MODE:-fixed_k1}"  # fixed_k1 | er

DIAG_SHIFT="${DIAG_SHIFT:-0.1}"
NOISE_STD="${NOISE_STD:-0.25}"
WEIGHT_LOW="${WEIGHT_LOW:-0.5}"
WEIGHT_HIGH="${WEIGHT_HIGH:-1.5}"
AS_TENSOR="${AS_TENSOR:-1}"

BATCH_SIZE="${BATCH_SIZE:-4}"
SHUFFLE="${SHUFFLE:-0}"

DATA_PATH="${DATA_PATH:-${SET_TYPE}/n_${NUM_NODES}_ndata_${N_DATA}_k0_${K0}_k1_${K1}_mode_${SAMPLE_MODE}_noise_${NOISE_STD}_size_${DATASET_SIZE}_seed_${RANDOM_STATE}.pt}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-}"

generate_dataset() {
  python unrolled_model/generate_dataset.py \
    --set_type "${SET_TYPE}" \
    --num_nodes "${NUM_NODES}" \
    --n_data "${N_DATA}" \
    --k0 "${K0}" \
    --k1 "${K1}" \
    --dataset_size "${DATASET_SIZE}" \
    --diag_shift "${DIAG_SHIFT}" \
    --noise_std "${NOISE_STD}" \
    --weight_low "${WEIGHT_LOW}" \
    --weight_high "${WEIGHT_HIGH}" \
    --as_tensor "${AS_TENSOR}" \
    --random_state "${RANDOM_STATE}" \
    --same_graph "${SAME_GRAPH}" \
    --sample_mode "${SAMPLE_MODE}" \
    --output "${DATA_PATH}"
}

inspect_dataset() {
  python unrolled_model/dataset.py "${DATA_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --shuffle "${SHUFFLE}"
}

train_model() {
  if [[ -z "${TRAIN_SCRIPT}" ]]; then
    echo "TRAIN_SCRIPT is empty. Set TRAIN_SCRIPT=path/to/train.py when the training entrypoint is ready." >&2
    exit 1
  fi
  python "${TRAIN_SCRIPT}" \
    --data_path "${DATA_PATH}" \
    --batch_size "${BATCH_SIZE}"
}

case "${MODE}" in
  generate)
    generate_dataset
    ;;
  inspect)
    inspect_dataset
    ;;
  all)
    generate_dataset
    inspect_dataset
    ;;
  train)
    train_model
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use generate, inspect, all, or train." >&2
    exit 1
    ;;
esac
