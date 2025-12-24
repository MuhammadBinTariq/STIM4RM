#!/bin/bash
set -euo pipefail

# ==========================================================
# ⚡ DEBUG MODE CONFIGURATION ⚡
# ==========================================================
MODEL_NAME="olmo_7b_instruct"
OUT_ROOT="../outputs_debug"        # Separate folder so we don't mess up real results
CKPT_ROOT="../checkpoints_debug"

# TINY SETTINGS FOR SPEED (5-10 mins total)
NUM_ITERS=2                        # Run 2 iters to prove the loop cycles correctly
NUM_ROLLOUTS=2                     # Minimum rollouts
BATCH_SIZE=1                       # Minimum batch
MAX_NEW_TOKENS=128                 # Super short generation (instant)

# GRPO KNOBS
LR="1e-6"
KL_BETA="0.04"
STIM_ALPHA="0.1"
GRPO_STEPS_PER_ITER=-1 

echo ">>> 🧪 STARTING DEBUG RUN..."

# 1. CREATE DUMMY DATA ON THE FLY
# We make a fake dataset with 2 simple math questions
mkdir -p ../data/applied
echo '[
  {"question": "What is 10 + 10?", "answer": "20"},
  {"question": "What is 5 * 5?", "answer": "25"}
]' > ../data/applied/debug_data.json

# 2. OVERRIDE TASK LIST TO USE DUMMY DATA
TASKS=(
    "applied:original:../data/applied/debug_data.json:../data/applied/examples.txt"
)

mkdir -p "${OUT_ROOT}" "${CKPT_ROOT}"
echo "Iteration,MAA,Accuracy,Mean_Rho" > "${OUT_ROOT}/metrics_log.csv"

# ==========================================================
# MAIN LOOP (Identical logic to real script)
# ==========================================================
for config in "${TASKS[@]}"; do
  IFS=":" read -r TASK_TYPE PT_TYPE DATA_PATH FEW_SHOT_PATH <<< "$config"

  EVAL_DIR="${OUT_ROOT}/${TASK_TYPE}/eval/${MODEL_NAME}"
  MEM_DIR="${OUT_ROOT}/${TASK_TYPE}/mem_score/${MODEL_NAME}"
  TRAIN_DIR="${OUT_ROOT}/${TASK_TYPE}/train/${MODEL_NAME}"
  mkdir -p "$EVAL_DIR" "${MEM_DIR}/local" "${TRAIN_DIR}"

  REF_MODEL="${MODEL_NAME}"
  CUR_POLICY="${MODEL_NAME}"

  for iter in $(seq 1 "${NUM_ITERS}"); do
    echo "--- DEBUG ITER ${iter}/${NUM_ITERS} ---"
    ITER_DIR="${TRAIN_DIR}/iter_${iter}"
    mkdir -p "${ITER_DIR}"

    # 1. INFERENCE
    echo ">>> Generating (Fast)..."
    # Note: We loop NUM_ROLLOUTS times
    for i in $(seq 1 "${NUM_ROLLOUTS}"); do
      python ./gen_eval/${TASK_TYPE}/inference.py \
        --model_name ${CUR_POLICY} \
        --data_path ${DATA_PATH} \
        --output_path "${ITER_DIR}/run_${i}.json" \
        --perturbation_type ${PT_TYPE} \
        --few_shot_path ${FEW_SHOT_PATH} \
        --batch_size ${BATCH_SIZE} \
        --max_new_tokens ${MAX_NEW_TOKENS} # Force short length
    done

    # Merge
    python -c "import json, glob; merged=[]; [merged.extend(json.load(open(p))) for p in glob.glob('${ITER_DIR}/run_*.json')]; json.dump(merged, open('${ITER_DIR}/merged_cot.json','w'), indent=2)"

    # 2. EVAL & REWARD
    echo ">>> PRM Scoring..."
    python get_reward.py --f_path "${ITER_DIR}/merged_cot.json" --output_path "${ITER_DIR}/merged_prm.json" --task_type ${TASK_TYPE}

    # Split
    python -c "import json; d=json.load(open('${ITER_DIR}/merged_prm.json')); c=[x for x in d if x.get('reward',0)>0.5]; w=[x for x in d if x.get('reward',0)<0.5]; json.dump(c,open('${ITER_DIR}/sampling_correct.json','w')); json.dump(w,open('${ITER_DIR}/sampling_wrong.json','w'))"

    # 3. STIM CALCULATION
    echo ">>> STIM Scoring..."
    for c in "correct" "wrong"; do
      if [ -s "${ITER_DIR}/sampling_${c}.json" ]; then
        python get_tokens.py --model_name ${CUR_POLICY} --f_path "${ITER_DIR}/sampling_${c}.json" --output_path "${ITER_DIR}/sampling_${c}_tokens.json" --task_type ${TASK_TYPE} --is_cpu
        python get_tokens.py --model_name ${CUR_POLICY} --f_path "${ITER_DIR}/sampling_${c}_tokens.json" --output_path "${ITER_DIR}/sampling_${c}_alter.json" --task_type ${TASK_TYPE}
        python cal_local.py --model_name ${MODEL_NAME} --f_path "${ITER_DIR}/sampling_${c}_alter.json" --output_path "${ITER_DIR}/${c}_score_local.json"
      else
        echo "[]" > "${ITER_DIR}/${c}_score_local.json"
      fi
    done

    # 4. METRICS
    echo ">>> Metrics..."
    python calc_maa.py --prm_file "${ITER_DIR}/merged_prm.json" --stim_correct "${ITER_DIR}/correct_score_local.json" --stim_wrong "${ITER_DIR}/wrong_score_local.json" --iter "${iter}" >> "${OUT_ROOT}/metrics_log.csv"

    # 5. GRPO UPDATE
    echo ">>> GRPO Update..."
    NEXT_CKPT="${CKPT_ROOT}/${TASK_TYPE}_iter_${iter}"
    mkdir -p "${NEXT_CKPT}"

    python grpo_update_stim.py \
      --policy_in "${CUR_POLICY}" \
      --policy_out "${NEXT_CKPT}" \
      --ref_model "${REF_MODEL}" \
      --merged_prm "${ITER_DIR}/merged_prm.json" \
      --stim_correct "${ITER_DIR}/correct_score_local.json" \
      --stim_wrong "${ITER_DIR}/wrong_score_local.json" \
      --num_rollouts "${NUM_ROLLOUTS}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --learning_rate "${LR}" \
      --kl_beta "${KL_BETA}" \
      --stim_alpha "${STIM_ALPHA}" \
      --steps "${GRPO_STEPS_PER_ITER}" \
      --batch_size "${BATCH_SIZE}"

    CUR_POLICY="${NEXT_CKPT}"
  done
done

echo ">>> ✅ DEBUG RUN COMPLETE! If you see this, you are ready for the real run."