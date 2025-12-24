#!/bin/bash
set -euo pipefail

# 1. APPLY PATCHES & INSTALL DEPS
export WANDB_MODE=disabled
export WANDB_SILENT=true
echo ">>> 🛡️ Applying Library Patches..."
python apply_patches.py

echo ">>> 📦 Checking Dependencies..."
pip install -q wandb ai2-olmo eval_type_backport

# ===============================
# 2. CONFIGURATION (for STIM + GRPO)
# ===============================
# MODEL_NAME="olmo_7b_instruct"          # starting SFT/policy
MODEL_NAME="allenai/OLMo-7B-Instruct"
OUT_ROOT="../outputs"
CKPT_ROOT="../checkpoints"            # where GRPO checkpoints go

NUM_ITERS=5                           # online GRPO iters per task
PROMPT_BATCH_SIZE=2     # <--- NEW: Number of Prompts per GRPO Update
NUM_ROLLOUTS=4                        # must match merge pattern in your STIM script
BATCH_SIZE=2

# GRPO knobs
LR="1e-6"
KL_BETA="0.04"                        # Increased slightly for stability
STIM_ALPHA="0.2"                      # The strength of the token-level reward
MAX_NEW_TOKENS=256
GRPO_STEPS_PER_ITER=-1                 # gradient steps per iter -> update on all the data generated
GRPO_PROMPT_BATCH=1                   # prompts per step (GRPO expands by NUM_ROLLOUTS)

echo "Starting Multi-Task STIM + GRPO (online) pipeline..."

# Format: "task_type:perturbation_type:data_file:few_shot_file"
TASKS=(
    "applied:original:../data/applied/original.json:../data/applied/examples.txt"
    # "formula:original:../data/formula/original.json:../data/formula/examples_cot.txt"
    # "cap:original:../data/cap/original.json:../data/cap/examples.txt"
)

mkdir -p "${OUT_ROOT}" "${CKPT_ROOT}"

# ==========================================================
# MAIN LOOP (tasks × iters)
# ==========================================================
for config in "${TASKS[@]}"; do
  IFS=":" read -r TASK_TYPE PT_TYPE DATA_PATH FEW_SHOT_PATH <<< "$config"

  echo "=================================================="
  echo "ONLINE TRAINING Task: $TASK_TYPE | Type: $PT_TYPE"
  echo "=================================================="

  # Per-task dirs (same structure as your script)
  EVAL_DIR="${OUT_ROOT}/${TASK_TYPE}/eval/${MODEL_NAME}"
  MEM_DIR="${OUT_ROOT}/${TASK_TYPE}/mem_score/${MODEL_NAME}"
  TRAIN_DIR="${OUT_ROOT}/${TASK_TYPE}/train/${MODEL_NAME}"
  mkdir -p "$EVAL_DIR" "${MEM_DIR}/local" "${MEM_DIR}/mid" "${MEM_DIR}/long" "${TRAIN_DIR}"

  # Reference model is the starting SFT (frozen)
  REF_MODEL="${MODEL_NAME}"

  # Current policy checkpoint (starts at SFT)
  CUR_POLICY="${MODEL_NAME}"

  for iter in $(seq 1 "${NUM_ITERS}"); do
    echo "--------------------------------------------------"
    echo "[${TASK_TYPE}] GRPO ITER ${iter}/${NUM_ITERS} | policy=${CUR_POLICY}"
    echo "--------------------------------------------------"

    ITER_DIR="${TRAIN_DIR}/iter_${iter}"
    mkdir -p "${ITER_DIR}"

    # --- SPLIT DATA INTO BATCHES ---
    echo ">>> Splitting dataset into batches of ${PROMPT_BATCH_SIZE}..."
    BATCH_DATA_DIR="${ITER_DIR}/batches_data"
    python split_data.py \
        --input "${DATA_PATH}" \
        --output_dir "${BATCH_DATA_DIR}" \
        --batch_size "${PROMPT_BATCH_SIZE}" \
        --shuffle

    # Get list of batch files
    BATCH_FILES=($(ls "${BATCH_DATA_DIR}"/batch_*.json | sort -V))
    
    # --- PROCESS EACH BATCH ---
    BATCH_IDX=0
    for BATCH_FILE in "${BATCH_FILES[@]}"; do
        BATCH_IDX=$((BATCH_IDX+1))
        BATCH_WORK_DIR="${ITER_DIR}/batch_${BATCH_IDX}"
        mkdir -p "${BATCH_WORK_DIR}"
        
        echo "--------------------------------------------------"
        echo "   ⚡ Processing Batch ${BATCH_IDX} (${PROMPT_BATCH_SIZE} prompts)"
        echo "      Policy: ${CUR_POLICY}"
        echo "--------------------------------------------------"
        # ------------------------------------------------
        # 1) INFERENCE (K rollouts) — SAME AS YOUR SCRIPT
        # ------------------------------------------------
        echo ">>> Generating ${NUM_ROLLOUTS} Rollouts..."
        for i in $(seq 1 "${NUM_ROLLOUTS}"); do
          echo "  - Rollout $i"

          # DATA_PATH -> BATCH_FILE
          if [ ! -f "${BATCH_WORK_DIR}/run_${i}.json" ]; then
            if [ "$TASK_TYPE" == "cap" ]; then
              python ./gen_eval/cap/inference.py \
                --model_name ${CUR_POLICY} \
                --few_shot_path ${FEW_SHOT_PATH} \
                --data_path ${BATCH_FILE} \
                --output_path "${BATCH_WORK_DIR}/run_${i}.json" \
                --prompt_type "cot" \
                --batch_size ${BATCH_SIZE}
            else
              python ./gen_eval/${TASK_TYPE}/inference.py \
                --model_name ${CUR_POLICY} \
                --data_path ${BATCH_FILE} \
                --output_path "${BATCH_WORK_DIR}/run_${i}.json" \
                --perturbation_type ${PT_TYPE} \
                --few_shot_path ${FEW_SHOT_PATH} \
                --batch_size ${BATCH_SIZE}
            fi
          fi
        done

    #     # Merge rollouts
    #     echo "  - Merging rollouts..."
    #     python -c "
    # import json, glob
    # merged=[]
    # for p in sorted(glob.glob('${ITER_DIR}/run_*.json')):
    #   with open(p) as f: merged.extend(json.load(f))
    # with open('${ITER_DIR}/merged_cot.json','w') as f: json.dump(merged,f,indent=2)
    # "

#         echo "  - Merging rollouts..."
# python - <<'PY'
# import json, glob, sys, os
# run_files = sorted(glob.glob(sys.argv[1]))
# out = sys.argv[2]
# merged=[]
# for p in run_files:
#     with open(p) as f: merged.extend(json.load(f))
# with open(out,'w') as f: json.dump(merged,f,indent=2)
# PY \
# "${BATCH_WORK_DIR}/run_*.json" \
# "${BATCH_WORK_DIR}/merged_cot.json"

        echo "  - Merging rollouts..."
        python -c '
        import json, glob, sys
        pattern = sys.argv[1]
        out = sys.argv[2]
        run_files = sorted(glob.glob(pattern))
        merged=[]
        for p in run_files:
            with open(p) as f: merged.extend(json.load(f))
        with open(out,"w") as f: json.dump(merged,f,indent=2)
        ' "${BATCH_WORK_DIR}/run_*.json" "${BATCH_WORK_DIR}/merged_cot.json"

        # ------------------------------------------------
        # 2) EVALUATION & PRM — SAME AS YOUR SCRIPT
        # ------------------------------------------------
        echo ">>> Evaluating..."
        if [ "$TASK_TYPE" == "cap" ]; then
          python ./gen_eval/cap/evaluate.py \
            --model_output_path "${BATCH_WORK_DIR}/merged_cot.json"
        else
          python ./gen_eval/${TASK_TYPE}/evaluate.py \
            --model_output_path "${BATCH_WORK_DIR}/merged_cot.json" \
            --perturbation_type ${PT_TYPE}
        fi

        echo ">>> Calculating PRM Scores..."
        python get_reward.py \
          --f_path "${BATCH_WORK_DIR}/merged_cot.json" \
          --output_path "${BATCH_WORK_DIR}/merged_prm.json" \
          --task_type ${TASK_TYPE}

        # ------------------------------------------------
        # 3) SPLIT correct/wrong — SAME AS YOUR SCRIPT
        # ------------------------------------------------
        python -c "
    import json
    with open('${BATCH_WORK_DIR}/merged_prm.json') as f:
      data=json.load(f)
    # Handle potential key variations for correctness
    correct=[d for d in data if d.get('is_correct', d.get('correct', 0))==1]
    wrong=[d for d in data if d.get('is_correct', d.get('correct', 0))==0]
    with open('${BATCH_WORK_DIR}/sampling_correct.json','w') as f: json.dump(correct,f,indent=2)
    with open('${BATCH_WORK_DIR}/sampling_wrong.json','w') as f: json.dump(wrong,f,indent=2)
    "

        # ------------------------------------------------
        # 4) TOKEN SELECTION & ALTS — SAME AS YOUR SCRIPT
        # ------------------------------------------------
        echo ">>> Selecting Tokens & Alternatives..."
        for c in "correct" "wrong"; do
          if [ -s "${BATCH_WORK_DIR}/sampling_${c}.json" ]; then
            python get_tokens.py \
              --model_name ${CUR_POLICY} \
              --f_path "${BATCH_WORK_DIR}/sampling_${c}.json" \
              --output_path "${BATCH_WORK_DIR}/sampling_${c}_tokens.json" \
              --task_type ${TASK_TYPE} \
              --is_cpu

            python get_tokens.py \
              --model_name ${CUR_POLICY} \
              --f_path "${BATCH_WORK_DIR}/sampling_${c}_tokens.json" \
              --output_path "${BATCH_WORK_DIR}/sampling_${c}_alter.json" \
              --task_type ${TASK_TYPE}
          fi
        done

        # ------------------------------------------------
        # 5) STIM LOCAL — SAME AS YOUR SCRIPT
        # ------------------------------------------------
        # echo ">>> Calculating STIM Scores..."
        # for c in "correct" "wrong"; do
        #   INPUT_FILE="${ITER_DIR}/sampling_${c}_alter.json"
        #   # Only run if file exists and has content
        #   if [ -s "$INPUT_FILE" ]; then
        #     echo "  - Local Score ($c)"
        #     python cal_local.py \
        #       --model_name ${CUR_POLICY} \
        #       --f_path "${INPUT_FILE}" \
        #       --output_path "${ITER_DIR}/${c}_score_local.json"
        #   else
        #     echo "  - Skipping ($c) - no data"
        #     # Create empty list to prevent file not found errors in python script
        #     echo "[]" > "${ITER_DIR}/${c}_score_local.json"
        #   fi
        # done

        echo ">>> Calculating STIM Scores..."
        for c in "correct" "wrong"; do
            INPUT_FILE="${BATCH_WORK_DIR}/sampling_${c}_alter.json"
            mkdir -p "${BATCH_WORK_DIR}/local"
            
            if [ -f "$INPUT_FILE" ] && [ -s "$INPUT_FILE" ]; then
                
                # --- LOCAL (Active) ---
                echo "  - Local Score ($c)"
                python cal_local.py \
                    --model_name ${MODEL_NAME} \
                    --f_path "${INPUT_FILE}" \
                    --output_path "${BATCH_WORK_DIR}/local/${c}_score.json"

                # --- MID-RANGE (Commented Out) ---
                # Uncomment below to generate Mid scores in JSON
                # echo "  - Mid Score ($c)"
                # python cal_mid.py \
                #     --f_path "${INPUT_FILE}" \
                #     --output_path "${MEM_DIR}/mid/mid_${c}_prefix.json" \
                #     --model_name ${MODEL_NAME}
                # python ./token_saliency/lerg_attr.py \
                #     --input_path "${MEM_DIR}/mid/mid_${c}_prefix.json" \
                #     --output_path "${MEM_DIR}/mid/mid_${c}_lerg.json" \
                #     --model_name ${MODEL_NAME} --mem_type mid
                # python cal_mid.py \
                #     --f_path "${MEM_DIR}/mid/mid_${c}_lerg.json" \
                #     --output_path "${MEM_DIR}/mid/${c}_score.json" \
                #     --model_name ${MODEL_NAME} --is_cpu

                # --- LONG-RANGE (Commented Out) ---
                # Uncomment below to generate Long scores in JSON
                # echo "  - Long Score ($c)"
                # python ./token_saliency/lerg_attr.py \
                #     --input_path "${INPUT_FILE}" \
                #     --output_path "${MEM_DIR}/long/long_${c}_lerg.json" \
                #     --model_name ${MODEL_NAME} --mem_type long
                # python cal_long.py \
                #     --f_path "${MEM_DIR}/long/long_${c}_lerg.json" \
                #     --output_path "${MEM_DIR}/long/${c}_score.json" \
                #     --model_name ${MODEL_NAME}
            else
                 # Create dummy empty file to prevent MAA/GRPO failure
                 echo "[]" > "${BATCH_WORK_DIR}/local/${c}_score.json"
            fi
        done

        # # ======================================================
        # # NEW: METRIC 1 (MAA) CALCULATION
        # # ======================================================
        # echo ">>> Calculating MAA..."
        # MAA_JSON=$(python calc_maa.py \
        #   --prm_file "${ITER_DIR}/merged_prm.json" \
        #   --stim_correct "${ITER_DIR}/correct_score_local.json" \
        #   --stim_wrong "${ITER_DIR}/wrong_score_local.json" \
        #   --iter "${iter}")
        
        # echo "METRICS: $MAA_JSON"
        # # Append to CSV for easy plotting later
        # echo "$MAA_JSON" | python -c "import sys, json; d=json.load(sys.stdin); print(f\"{d['iter']},{d['MAA']},{d['Accuracy']},{d['Mean_Rho']}\")" >> "${OUT_ROOT}/metrics_log.csv"

        # ------------------------------------------------
        # 6) GRPO UPDATE (Using Custom STIM Trainer)
        # ------------------------------------------------
        NEXT_CKPT="${CKPT_ROOT}/${TASK_TYPE}_iter_${iter}_batch_${BATCH_IDX}"
        mkdir -p "${NEXT_CKPT}"

        echo ">>> GRPO Update (policy -> ${NEXT_CKPT})"
        
        # We call our custom python script here
        python grpo_update_stim.py \
          --policy_in "${CUR_POLICY}" \
          --policy_out "${NEXT_CKPT}" \
          --ref_model "${REF_MODEL}" \
          --merged_prm "${BATCH_WORK_DIR}/merged_prm.json" \
          --stim_correct "${BATCH_WORK_DIR}/local/correct_score.json" \
          --stim_wrong "${BATCH_WORK_DIR}/local/wrong_score.json" \
          --num_rollouts "${NUM_ROLLOUTS}" \
          --max_new_tokens "${MAX_NEW_TOKENS}" \
          --learning_rate "${LR}" \
          --kl_beta "${KL_BETA}" \
          --stim_alpha "${STIM_ALPHA}" \
          --steps "${GRPO_STEPS_PER_ITER}" \
          --batch_size "${BATCH_SIZE}"
        #   --batch_size "${GRPO_PROMPT_BATCH}"

        CUR_POLICY="${NEXT_CKPT}"
      done


    # =======================
    # MERGE BATCH RESULTS FOR MAA (PER ITER)
    # =======================
    echo ">>> Merging batch PRM + STIM results for iter ${iter} (for MAA only)..."

    python - <<PY
import json, glob, os

iter_dir = r"${ITER_DIR}"
os.makedirs(iter_dir, exist_ok=True)

def merge(pattern, out_path):
    merged = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            obj = json.load(f)
            if isinstance(obj, list):
                merged.extend(obj)
            else:
                merged.append(obj)
    with open(out_path,"w") as f:
        json.dump(merged, f, indent=2)

# merged_prm across batches
merge(os.path.join(iter_dir, "batch_*", "merged_prm.json"),
      os.path.join(iter_dir, "merged_prm.json"))

# STIM: correct & wrong
merge(os.path.join(iter_dir, "batch_*", "local", "correct_score.json"),
      os.path.join(iter_dir, "correct_score_local.json"))
merge(os.path.join(iter_dir, "batch_*", "local", "wrong_score.json"),
      os.path.join(iter_dir, "wrong_score_local.json"))
PY

    # ======================================================
    # MAA CALCULATION (ONCE PER ITER)
    # ======================================================
    echo ">>> Calculating MAA for iter ${iter}..."
    MAA_JSON=$(python calc_maa.py \
      --prm_file "${ITER_DIR}/merged_prm.json" \
      --stim_correct "${ITER_DIR}/correct_score_local.json" \
      --stim_wrong "${ITER_DIR}/wrong_score_local.json" \
      --iter "${iter}")

    echo "METRICS: $MAA_JSON"
    echo "$MAA_JSON" | python -c \
"import sys, json; d=json.load(sys.stdin); print(f\"${iter},ALL,{d['MAA']},{d['Accuracy']},{d['Mean_Rho']}\")" \
      >> "${OUT_ROOT}/metrics_log.csv"

done

echo "[DONE] Multi-task STIM + GRPO finished."

# ==========================================================
# METRIC 3: ROBUSTNESS EVALUATION (End of Run)
# ==========================================================
echo ">>> RUNNING OOD ROBUSTNESS EVALUATION <<<"

OOD_TASKS=(
    "applied:base2:../data/applied/changing_base.json"
)

for config in "${OOD_TASKS[@]}"; do
  IFS=":" read -r TASK_TYPE PT_TYPE DATA_PATH <<< "$config"
  
  # Define where to save this specific evaluation
  # Re-define EVAL_DIR here since we are outside the main loop
  FINAL_EVAL_DIR="${OUT_ROOT}/${TASK_TYPE}/eval/${MODEL_NAME}"
  mkdir -p "${FINAL_EVAL_DIR}"

  # FINAL_CKPT="${CKPT_ROOT}/${TASK_TYPE}_iter_${NUM_ITERS}" 
  FINAL_CKPT="${CUR_POLICY}"

  FINAL_EVAL_DIR="${OUT_ROOT}/${TASK_TYPE}/eval_final"
  mkdir -p "${FINAL_EVAL_DIR}"
  
  if [ -d "$FINAL_CKPT" ]; then
      echo "Testing ${FINAL_CKPT} on ${PT_TYPE}..."
      
      # SAVE TO EVAL_DIR (Cleaner)
      python ./gen_eval/${TASK_TYPE}/inference.py \
          --model_name ${FINAL_CKPT} \
          --data_path ${DATA_PATH} \
          --output_path "${FINAL_EVAL_DIR}/ood_eval_${PT_TYPE}.json" \
          --perturbation_type ${PT_TYPE} \
          --batch_size ${BATCH_SIZE}
      
      echo "Robustness Score for ${PT_TYPE}:"
      python ./gen_eval/${TASK_TYPE}/evaluate.py \
          --model_output_path "${FINAL_EVAL_DIR}/ood_eval_${PT_TYPE}.json" \
          --perturbation_type ${PT_TYPE}
  fi
done

echo "[DONE] Pipeline Finished."