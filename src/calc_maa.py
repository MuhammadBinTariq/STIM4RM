import json
import numpy as np
import argparse
import hashlib
import os

def stable_hash(text):
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prm_file", required=True)
    parser.add_argument("--stim_correct", required=True)
    parser.add_argument("--stim_wrong", required=True)
    parser.add_argument("--iter", type=int, required=True)
    args = parser.parse_args()

    # 1. Load STIM Scores (The Memorization Penalty)
    stim_map = {}
    for fpath in [args.stim_correct, args.stim_wrong]:
        if not os.path.exists(fpath): continue
        try:
            with open(fpath) as f: 
                data = json.load(f)
                # Handle {"data": [...]} wrapper if present
                if isinstance(data, dict) and "data" in data: data = data["data"]
                
                for item in data:
                    p = item.get("prompt", item.get("question", item.get("input", "")))
                    c = item.get("model_output", item.get("completion", item.get("response", "")))
                    
                    # Extract mean STIM (rho) for this sequence
                    # Usually found in 'token_alternative_fre' or 'token_scores'
                    scores = item.get("token_alternative_fre", item.get("token_scores", []))
                    vals = [x.get('corr', x.get('score', 0)) for x in scores if isinstance(x, dict)]
                    
                    if vals and p and c:
                        mean_rho = sum(vals) / len(vals)
                        stim_map[stable_hash(p+c)] = mean_rho
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    # 2. Load PRM (Correctness) & Calculate MAA
    try:
        with open(args.prm_file) as f: prm_data = json.load(f)
    except:
        print("Error: Could not load PRM file.")
        return

    total_maa = 0.0
    total_acc = 0.0
    valid_count = 0
    rhos = []

    for item in prm_data:
        p = item.get("prompt", item.get("question", ""))
        c = item.get("model_output", item.get("completion", ""))
        
        # 1.0 if correct, 0.0 if wrong
        r_val = float(item.get("reward", item.get("prm_score", 0)))
        is_correct = 1.0 if r_val > 0.5 else 0.0
        
        # Get Penalty
        h = stable_hash(p+c)
        rho = stim_map.get(h, 0.0) # Default to 0 penalty if missing
        rhos.append(rho)
        
        # MAA Formula: Correctness * (1 - Memorization)
        maa_contribution = is_correct * (1.0 - rho)
        
        total_maa += maa_contribution
        total_acc += is_correct
        valid_count += 1

    final_maa = total_maa / max(1, valid_count)
    final_acc = total_acc / max(1, valid_count)
    mean_rho = sum(rhos) / max(1, len(rhos))

    # Print JSON for the Bash script to capture
    print(f'{{"iter": {args.iter}, "MAA": {final_maa:.4f}, "Accuracy": {final_acc:.4f}, "Mean_Rho": {mean_rho:.4f}}}')

if __name__ == "__main__":
    import os
    main()