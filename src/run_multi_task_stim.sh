#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME="olmo_7b_instruct"
OUT_ROOT="../outputs"

echo "Starting Multi-Task STIM Pipeline..."

# Define tasks and their specific configurations
# Format: "task_type:perturbation_type:data_file:few_shot_file"
TASKS=(
    "applied:original:../data/applied/original.json:../data/applied/examples.txt"
    "formula:original:../data/formula/original.json:../data/formula/examples_cot.txt"
    "cap:cot:../data/cap/book_title.json:../data/cap/examples.json" 
)

for config in "${TASKS[@]}"; do
    IFS=":" read -r TASK_TYPE PT_TYPE DATA_PATH FEW_SHOT_PATH <<< "$config"
    
    echo "=================================================="
    echo "Processing Task: $TASK_TYPE | Type: $PT_TYPE"
    echo "=================================================="

    # Setup directories
    EVAL_DIR="${OUT_ROOT}/${TASK_TYPE}/eval/${MODEL_NAME}"
    MEM_DIR="${OUT_ROOT}/${TASK_TYPE}/mem_score/${MODEL_NAME}"
    mkdir -p "$EVAL_DIR" "${MEM_DIR}/local" "${MEM_DIR}/mid" "${MEM_DIR}/long"

    # ------------------------------------------------
    # 1. INFERENCE (4 Rollouts)
    # ------------------------------------------------
    echo ">>> Generating 4 Rollouts..."
    for i in {1..4}; do
        echo "  - Rollout $i"
        
        # Handle arguments based on task type
        if [ "$TASK_TYPE" == "cap" ]; then
            # Capitalization uses different flags
             python ./gen_eval/cap/inference.py \
                --model_name ${MODEL_NAME} \
                --few_shot_path ${FEW_SHOT_PATH} \
                --data_path ${DATA_PATH} \
                --output_path "${EVAL_DIR}/run_${i}.json" \
                --prompt_type "cot" \
                --batch_size 2
        else
            # Applied and Formula share similar flags
            python ./gen_eval/${TASK_TYPE}/inference.py \
                --model_name ${MODEL_NAME} \
                --data_path ${DATA_PATH} \
                --output_path "${EVAL_DIR}/run_${i}.json" \
                --perturbation_type ${PT_TYPE} \
                --few_shot_path ${FEW_SHOT_PATH} \
                --batch_size 2
        fi
    done

    # Merge rollouts
    echo "  - Merging rollouts..."
    python -c "
import json
merged = []
for i in range(1, 5):
    try:
        with open('${EVAL_DIR}/run_' + str(i) + '.json') as f:
            merged.extend(json.load(f))
    except: pass
with open('${EVAL_DIR}/merged_cot.json', 'w') as f:
    json.dump(merged, f, indent=2)
"

    # ------------------------------------------------
    # 2. EVALUATION & REWARD
    # ------------------------------------------------
    echo ">>> Evaluating..."
    
    # 2a. Correctness
    if [ "$TASK_TYPE" == "cap" ]; then
         python ./gen_eval/cap/evaluate.py --model_output_path "${EVAL_DIR}/merged_cot.json"
    else
         python ./gen_eval/${TASK_TYPE}/evaluate.py \
            --model_output_path "${EVAL_DIR}/merged_cot.json" \
            --perturbation_type ${PT_TYPE}
    fi

    # 2b. Process Reward Model (PRM)
    echo ">>> Calculating PRM Scores..."
    python get_reward.py \
        --f_path "${EVAL_DIR}/merged_cot.json" \
        --output_path "${EVAL_DIR}/merged_prm.json" \
        --task_type ${TASK_TYPE}

    # ------------------------------------------------
    # 3. SPLIT & PREPARE FOR STIM
    # ------------------------------------------------
    # Split into correct/wrong for token analysis
    python -c "
import json
with open('${EVAL_DIR}/merged_prm.json') as f:
    data = json.load(f)
correct = [d for d in data if d.get('is_correct', 0) == 1]
wrong = [d for d in data if d.get('is_correct', 0) == 0]
with open('${EVAL_DIR}/sampling_correct.json', 'w') as f:
    json.dump(correct, f, indent=2)
with open('${EVAL_DIR}/sampling_wrong.json', 'w') as f:
    json.dump(wrong, f, indent=2)
"

    # ------------------------------------------------
    # 4. TOKEN SELECTION & ALTERNATIVES
    # ------------------------------------------------
    echo ">>> Selecting Tokens & Alternatives..."
    for c in "correct" "wrong"; do
        if [ -s "${EVAL_DIR}/sampling_${c}.json" ]; then
            # Select Tokens (CPU)
            python get_tokens.py \
                --model_name ${MODEL_NAME} \
                --f_path "${EVAL_DIR}/sampling_${c}.json" \
                --output_path "${EVAL_DIR}/sampling_${c}_tokens.json" \
                --task_type ${TASK_TYPE} \
                --is_cpu

            # Calculate Alternatives (GPU)
            python get_tokens.py \
                --model_name ${MODEL_NAME} \
                --f_path "${EVAL_DIR}/sampling_${c}_tokens.json" \
                --output_path "${EVAL_DIR}/sampling_${c}_alter.json" \
                --task_type ${TASK_TYPE}
        fi
    done

    # ------------------------------------------------
    # 5. STIM SCORE CALCULATION
    # ------------------------------------------------
    echo ">>> Calculating STIM Scores..."
    for c in "correct" "wrong"; do
        INPUT_FILE="${EVAL_DIR}/sampling_${c}_alter.json"
        
        if [ -f "$INPUT_FILE" ]; then
            
            # --- LOCAL (Active) ---
            echo "  - Local Score ($c)"
            python cal_local.py \
                --model_name ${MODEL_NAME} \
                --f_path "${INPUT_FILE}" \
                --output_path "${MEM_DIR}/local/${c}_score.json"

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

        fi
    done
done