# #!/bin/bash
# set -euo pipefail

# # ==========================================================
# # ⚡ DEBUG MODE CONFIGURATION ⚡
# # ==========================================================
# MODEL_NAME="olmo_7b_instruct"
# OUT_ROOT="../outputs_debug"        # Separate folder so we don't mess up real results
# CKPT_ROOT="../checkpoints_debug"

# # TINY SETTINGS FOR SPEED (5-10 mins total)
# NUM_ITERS=2                        # Run 2 iters to prove the loop cycles correctly
# NUM_ROLLOUTS=2                     # Minimum rollouts
# BATCH_SIZE=1                       # Minimum batch
# MAX_NEW_TOKENS=128                 # Super short generation (instant)

# # GRPO KNOBS
# LR="1e-6"
# KL_BETA="0.04"
# STIM_ALPHA="0.1"
# GRPO_STEPS_PER_ITER=-1 

# echo ">>> 🧪 STARTING DEBUG RUN..."

# # 1. CREATE DUMMY DATA ON THE FLY
# # We make a fake dataset with 2 simple math questions
# mkdir -p ../data/applied
# echo '[
#   {"question": "What is 10 + 10?", "answer": "20"},
#   {"question": "What is 5 * 5?", "answer": "25"}
# ]' > ../data/applied/debug_data.json

# # 2. OVERRIDE TASK LIST TO USE DUMMY DATA
# TASKS=(
#     "applied:original:../data/applied/debug_data.json:../data/applied/examples.txt"
# )

# mkdir -p "${OUT_ROOT}" "${CKPT_ROOT}"
# echo "Iteration,MAA,Accuracy,Mean_Rho" > "${OUT_ROOT}/metrics_log.csv"

# # ==========================================================
# # MAIN LOOP (Identical logic to real script)
# # ==========================================================
# for config in "${TASKS[@]}"; do
#   IFS=":" read -r TASK_TYPE PT_TYPE DATA_PATH FEW_SHOT_PATH <<< "$config"

#   EVAL_DIR="${OUT_ROOT}/${TASK_TYPE}/eval/${MODEL_NAME}"
#   MEM_DIR="${OUT_ROOT}/${TASK_TYPE}/mem_score/${MODEL_NAME}"
#   TRAIN_DIR="${OUT_ROOT}/${TASK_TYPE}/train/${MODEL_NAME}"
#   mkdir -p "$EVAL_DIR" "${MEM_DIR}/local" "${TRAIN_DIR}"

#   REF_MODEL="${MODEL_NAME}"
#   CUR_POLICY="${MODEL_NAME}"

#   for iter in $(seq 1 "${NUM_ITERS}"); do
#     echo "--- DEBUG ITER ${iter}/${NUM_ITERS} ---"
#     ITER_DIR="${TRAIN_DIR}/iter_${iter}"
#     mkdir -p "${ITER_DIR}"

#     # 1. INFERENCE
#     echo ">>> Generating (Fast)..."
#     # Note: We loop NUM_ROLLOUTS times
#     for i in $(seq 1 "${NUM_ROLLOUTS}"); do
#       python ./gen_eval/${TASK_TYPE}/inference.py \
#         --model_name ${CUR_POLICY} \
#         --data_path ${DATA_PATH} \
#         --output_path "${ITER_DIR}/run_${i}.json" \
#         --perturbation_type ${PT_TYPE} \
#         --few_shot_path ${FEW_SHOT_PATH} \
#         --batch_size ${BATCH_SIZE} \
#         --max_new_tokens ${MAX_NEW_TOKENS} # Force short length
#     done

#     # Merge
#     python -c "import json, glob; merged=[]; [merged.extend(json.load(open(p))) for p in glob.glob('${ITER_DIR}/run_*.json')]; json.dump(merged, open('${ITER_DIR}/merged_cot.json','w'), indent=2)"

#     # 1.5 EVALUATION (Crucial Fix: Adds 'is_correct' to the JSON)
#     echo ">>> Evaluating..."
#     python ./gen_eval/${TASK_TYPE}/evaluate.py \
#       --model_output_path "${ITER_DIR}/merged_cot.json" \
#       --perturbation_type ${PT_TYPE}

#     # 2. EVAL & REWARD
#     echo ">>> PRM Scoring..."
#     python get_reward.py --f_path "${ITER_DIR}/merged_cot.json" --output_path "${ITER_DIR}/merged_prm.json" --task_type ${TASK_TYPE}

#     # Split
#     python -c "import json; d=json.load(open('${ITER_DIR}/merged_prm.json')); c=[x for x in d if x.get('reward',0)>0.5]; w=[x for x in d if x.get('reward',0)<0.5]; json.dump(c,open('${ITER_DIR}/sampling_correct.json','w')); json.dump(w,open('${ITER_DIR}/sampling_wrong.json','w'))"

#     echo ">>> Selecting Tokens & Alternatives..."
#     for c in "correct" "wrong"; do
#       if [ -s "${ITER_DIR}/sampling_${c}.json" ]; then
#         python get_tokens.py \
#           --model_name ${CUR_POLICY} \
#           --f_path "${ITER_DIR}/sampling_${c}.json" \
#           --output_path "${ITER_DIR}/sampling_${c}_tokens.json" \
#           --task_type ${TASK_TYPE} \
#           --is_cpu

#         python get_tokens.py \
#           --model_name ${CUR_POLICY} \
#           --f_path "${ITER_DIR}/sampling_${c}_tokens.json" \
#           --output_path "${ITER_DIR}/sampling_${c}_alter.json" \
#           --task_type ${TASK_TYPE}
#       fi
#     done

#     # 3. STIM CALCULATION
#     echo ">>> STIM Scoring..."
#     for c in "correct" "wrong"; do
#       if [ -s "${ITER_DIR}/sampling_${c}.json" ]; then
#         python get_tokens.py --model_name ${CUR_POLICY} --f_path "${ITER_DIR}/sampling_${c}.json" --output_path "${ITER_DIR}/sampling_${c}_tokens.json" --task_type ${TASK_TYPE} --is_cpu
#         python get_tokens.py --model_name ${CUR_POLICY} --f_path "${ITER_DIR}/sampling_${c}_tokens.json" --output_path "${ITER_DIR}/sampling_${c}_alter.json" --task_type ${TASK_TYPE}
#         python cal_local.py --model_name ${MODEL_NAME} --f_path "${ITER_DIR}/sampling_${c}_alter.json" --output_path "${ITER_DIR}/${c}_score_local.json"
#       else
#         echo "[]" > "${ITER_DIR}/${c}_score_local.json"
#       fi
#     done

#     # 4. METRICS
#     echo ">>> Metrics..."
#     python calc_maa.py --prm_file "${ITER_DIR}/merged_prm.json" --stim_correct "${ITER_DIR}/correct_score_local.json" --stim_wrong "${ITER_DIR}/wrong_score_local.json" --iter "${iter}" >> "${OUT_ROOT}/metrics_log.csv"

#     # 5. GRPO UPDATE
#     echo ">>> GRPO Update..."
#     NEXT_CKPT="${CKPT_ROOT}/${TASK_TYPE}_iter_${iter}"
#     mkdir -p "${NEXT_CKPT}"

#     python grpo_update_stim.py \
#       --policy_in "${CUR_POLICY}" \
#       --policy_out "${NEXT_CKPT}" \
#       --ref_model "${REF_MODEL}" \
#       --merged_prm "${ITER_DIR}/merged_prm.json" \
#       --stim_correct "${ITER_DIR}/correct_score_local.json" \
#       --stim_wrong "${ITER_DIR}/wrong_score_local.json" \
#       --num_rollouts "${NUM_ROLLOUTS}" \
#       --max_new_tokens "${MAX_NEW_TOKENS}" \
#       --learning_rate "${LR}" \
#       --kl_beta "${KL_BETA}" \
#       --stim_alpha "${STIM_ALPHA}" \
#       --steps "${GRPO_STEPS_PER_ITER}" \
#       --batch_size "${BATCH_SIZE}"

#     CUR_POLICY="${NEXT_CKPT}"
#   done
# done

# echo ">>> ✅ DEBUG RUN COMPLETE! If you see this, you are ready for the real run."

set -euo pipefail

# ==========================================================
# 🛑 STRICT DEBUG MODE (NO SKIPPING) 🛑
# ==========================================================
MODEL_NAME="olmo_7b_instruct"
OUT_ROOT="../outputs_debug"
CKPT_ROOT="../checkpoints_debug"

# CONFIG
NUM_ITERS=2
NUM_ROLLOUTS=2
BATCH_SIZE=1
MAX_NEW_TOKENS=128
LR="1e-6"
KL_BETA="0.04"
STIM_ALPHA="0.1"
GRPO_STEPS_PER_ITER=-1 

echo ">>> 🛑 STARTING STRICT DEBUG RUN..."

# 1. PREPARE DATA
echo ">>> [Step 1] Creating Debug Data..."
mkdir -p ../data/applied
# We grab 2 real examples.
# python -c "import json; data=json.load(open('../data/applied/original.json')); json.dump(data[:2], open('../data/applied/debug_data.json','w'))"
# ls -l ../data/applied/debug_data.json

# TASKS=("applied:original:../data/applied/debug_data.json:../data/applied/examples.txt")
TASKS=(
    "applied:original:../data/applied/original.json:../data/applied/examples.txt"
    # "formula:original:../data/formula/original.json:../data/formula/examples_cot.txt"
    # "cap:original:../data/cap/original.json:../data/cap/examples.txt"
)
mkdir -p "${OUT_ROOT}" "${CKPT_ROOT}"

# ==========================================================
# MAIN LOOP
# ==========================================================
for config in "${TASKS[@]}"; do
  IFS=":" read -r TASK_TYPE PT_TYPE DATA_PATH FEW_SHOT_PATH <<< "$config"

  EVAL_DIR="${OUT_ROOT}/${TASK_TYPE}/eval/${MODEL_NAME}"
  TRAIN_DIR="${OUT_ROOT}/${TASK_TYPE}/train/${MODEL_NAME}"
  mkdir -p "$EVAL_DIR" "${TRAIN_DIR}"

  REF_MODEL="${MODEL_NAME}"
  CUR_POLICY="${MODEL_NAME}"

  for iter in $(seq 1 "${NUM_ITERS}"); do
    echo "=================================================="
    echo "ITERATION ${iter}"
    echo "=================================================="
    ITER_DIR="${TRAIN_DIR}/iter_${iter}"
    mkdir -p "${ITER_DIR}"

    # -------------------------------------------------
    # 1. INFERENCE
    # -------------------------------------------------
    echo ">>> [Step 1] Running Inference..."
    for i in $(seq 1 "${NUM_ROLLOUTS}"); do
        python ./gen_eval/${TASK_TYPE}/inference.py \
          --model_name ${CUR_POLICY} \
          --data_path ${DATA_PATH} \
          --output_path "${ITER_DIR}/run_${i}.json" \
          --perturbation_type ${PT_TYPE} \
          --few_shot_path ${FEW_SHOT_PATH} \
          --batch_size ${BATCH_SIZE} \
          --max_new_tokens ${MAX_NEW_TOKENS}
    done

    echo ">>> [Step 1.5] Merging..."
    python -c "import json, glob; merged=[]; [merged.extend(json.load(open(p))) for p in glob.glob('${ITER_DIR}/run_*.json')]; json.dump(merged, open('${ITER_DIR}/merged_cot.json','w'), indent=2)"
    ls -l "${ITER_DIR}/merged_cot.json"

    # -------------------------------------------------
    # 2. EVALUATION
    # -------------------------------------------------
    echo ">>> [Step 2] Evaluating (Adding 'is_correct')..."
    python ./gen_eval/${TASK_TYPE}/evaluate.py \
      --model_output_path "${ITER_DIR}/merged_cot.json" \
      --perturbation_type ${PT_TYPE}
    
    # -------------------------------------------------
    # 3. REWARD & SPLIT
    # -------------------------------------------------
    echo ">>> [Step 3] PRM Scoring & Splitting..."
    python get_reward.py \
      --f_path "${ITER_DIR}/merged_cot.json" \
      --output_path "${ITER_DIR}/merged_prm.json" \
      --task_type ${TASK_TYPE}

    python -c "import json; d=json.load(open('${ITER_DIR}/merged_prm.json')); c=[x for x in d if x.get('is_correct', x.get('correct', 0))==1]; w=[x for x in d if x.get('is_correct', x.get('correct', 0))==0]; json.dump(c,open('${ITER_DIR}/sampling_correct.json','w')); json.dump(w,open('${ITER_DIR}/sampling_wrong.json','w'))"
    
    echo "    Debug Split Stats:"
    ls -l "${ITER_DIR}/sampling_correct.json"
    ls -l "${ITER_DIR}/sampling_wrong.json"

    # -------------------------------------------------
    # 4. STIM (The Part That Was Skipping)
    # -------------------------------------------------
    echo ">>> [Step 4] STIM Token Calculation..."
    
    for c in "correct" "wrong"; do
        INPUT="${ITER_DIR}/sampling_${c}.json"
        TOKENS="${ITER_DIR}/sampling_${c}_tokens.json"
        ALTER="${ITER_DIR}/sampling_${c}_alter.json"
        SCORE="${ITER_DIR}/${c}_score_local.json"

        # Check if input has data
        COUNT=$(python -c "import json; print(len(json.load(open('$INPUT'))))")
        echo "    Processing '$c' list (Size: $COUNT items)"

        if [ "$COUNT" -eq 0 ]; then
            echo "    -> List is empty. Creating empty score file to prevent downstream crash."
            echo "[]" > "$SCORE"
            continue
        fi

        # FORCE RUN: Extract Tokens
        echo "    -> Running get_tokens.py (Extract)..."
        python get_tokens.py \
            --model_name ${CUR_POLICY} \
            --f_path "$INPUT" \
            --output_path "$TOKENS" \
            --task_type ${TASK_TYPE} \
            --is_cpu

        if [ ! -s "$TOKENS" ]; then
            echo "CRITICAL ERROR: $TOKENS is empty or missing! get_tokens.py failed to extract content."
            exit 1
        fi

        # FORCE RUN: Get Alternatives
        echo "    -> Running get_tokens.py (Alternatives)..."
        python get_tokens.py \
            --model_name ${CUR_POLICY} \
            --f_path "$TOKENS" \
            --output_path "$ALTER" \
            --task_type ${TASK_TYPE}

        if [ ! -s "$ALTER" ]; then
            echo "CRITICAL ERROR: $ALTER is empty! get_tokens.py failed to find alternatives."
            exit 1
        fi

        # FORCE RUN: Calculate Local Score
        echo "    -> Running cal_local.py..."
        python cal_local.py \
            --model_name ${MODEL_NAME} \
            --f_path "$ALTER" \
            --output_path "$SCORE"
    done

    # -------------------------------------------------
    # 5. METRICS & GRPO
    # -------------------------------------------------
    echo ">>> [Step 5] GRPO Update..."
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

echo ">>> ✅ STRICT DEBUG RUN COMPLETE!"