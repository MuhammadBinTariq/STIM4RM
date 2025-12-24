# import argparse
# import json
# import os
# import torch
# import hashlib
# import wandb
# from typing import List, Dict, Any
# from dataclasses import dataclass
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # TRL Imports
# from trl import GRPOTrainer, GRPOConfig
# import transformers
# from hf_olmo import OLMoForCausalLM
# # TRICK: Manually add OLMo class to transformers module to satisfy TRL check
# setattr(transformers, "OLMoForCausalLM", OLMoForCausalLM)

import argparse
import json
import os
import torch
import hashlib
import wandb
from typing import List, Dict, Any
from dataclasses import dataclass
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 0. REGISTER OLMO CLASS ---
try:
    from hf_olmo import OLMoForCausalLM
    setattr(transformers, "OLMoForCausalLM", OLMoForCausalLM)
except ImportError:
    print("Warning: hf_olmo not installed.")

# 3. NOW import TRL (It will now see the class exists in transformers)
from trl import GRPOTrainer, GRPOConfig

# ==============================================================================
# 1. HELPER FUNCTIONS (Data Loading & Hashing)
# ==============================================================================

def stable_hash(text: str) -> str:
    """Creates a hash of the text to reliably map prompts/completions to scores."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

def load_json(path: str) -> List[Dict]:
    """Robust JSON loader that handles empty files or different list formats."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list): return data
            # Handle cases where data is wrapped in {"data": [...]}
            if isinstance(data, dict): return data.get("data", [])
            return []
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return []

def load_stim_scores(path: str) -> Dict[str, Dict[int, float]]:
    """
    Loads STIM local scores.
    Returns: { hash(prompt + completion): { token_index: score } }
    """
    data = load_json(path)
    stim_map = {}
    
    for item in data:
        # Extract fields (handle various key naming conventions)
        prompt = item.get("prompt", item.get("question", item.get("input", "")))
        # Some STIM scripts nest the original item in 'data'
        if not prompt and "data" in item and isinstance(item["data"], dict):
            prompt = item["data"].get("prompt", "")
            
        completion = item.get("model_output", item.get("completion", item.get("response", "")))
        
        # Depending on your cal_local.py, scores might be in 'token_scores' or 'token_alternative_fre'
        # We assume a list of dicts: [{'pos': 12, 'corr': 0.5}, ...]
        scores = item.get("token_alternative_fre", item.get("token_scores", []))
        
        if prompt and completion and scores:
            key = stable_hash(prompt + completion)
            token_map = {}
            
            # Parse the score list
            for s in scores:
                # Handle cases where s is a list or dict
                if isinstance(s, dict):
                    pos = s.get("pos", s.get("position"))
                    val = s.get("corr", s.get("score", s.get("stim")))
                else:
                    continue # Skip malformed
                
                if pos is not None and val is not None:
                    token_map[int(pos)] = float(val)
            
            stim_map[key] = token_map
            
    return stim_map

# ==============================================================================
# 2. CUSTOM TRAINER (The Core Logic)
# ==============================================================================

class STIMGRPOTrainer(GRPOTrainer):
    """
    Subclass of GRPOTrainer that adds Token-Level Reward Shaping (STIM).
    """
    def __init__(self, stim_alpha: float, stim_data: Dict[str, Dict[int, float]], tokenizer, **kwargs):
        if 'ref_model' in kwargs:
            del kwargs['ref_model']
        super().__init__(**kwargs)
        self.stim_alpha = stim_alpha
        self.stim_data = stim_data # The look-up table for scores
        self.processing_class = tokenizer # Store tokenizer for decoding in loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        We override compute_loss to inject STIM scores into the advantages.
        """
        # 1. Standard Data Extraction from TRL inputs
        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        prompt_mask = inputs.get("prompt_mask")
        completion_mask = inputs.get("completion_mask")
        
        # TRL calculates sequence-level advantages (A) for us
        # Shape: (Batch, Group_Size) -> Flatten to (Batch * Group)
        advantages = inputs["advantages"].flatten()

        # Calculate Mean Task Advantage (Before Penalty)
        mean_task_adv = advantages.mean().detach()
        
        # 2. Forward Pass (Current Policy) to get per-token logprobs
        # We need to reconstruct input_ids (Prompt + Completion)
        B, G, L = completion_ids.shape
        flat_prompts = prompt_ids.repeat_interleave(G, dim=0) 
        flat_completions = completion_ids.flatten(0, 1)
        
        input_ids = torch.cat([flat_prompts, flat_completions], dim=1)
        
        if prompt_mask is not None:
            flat_p_mask = prompt_mask.repeat_interleave(G, dim=0)
            flat_c_mask = completion_mask.flatten(0, 1)
            attention_mask = torch.cat([flat_p_mask, flat_c_mask], dim=1)
        else:
            attention_mask = None

        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Extract logits for the completion part only
        # logits shape: (B*G, Total_Len, Vocab)
        # We shift by 1 for next-token prediction
        logits = outputs.logits[:, :-1, :]
        prompt_len = prompt_ids.shape[1]
        
        # Slice out the completion logits
        # completion starts at prompt_len. 
        completion_logits = logits[:, prompt_len-1 : prompt_len-1+L, :]
        
        # Calculate Log Softmax
        per_token_logps = torch.gather(
            torch.nn.functional.log_softmax(completion_logits, dim=-1),
            dim=2,
            index=flat_completions.unsqueeze(2)
        ).squeeze(2)

        # 3. Calculate Ratio (Policy / Ref)
        # TRL usually passes 'old_logps' (from rollout)
        if "old_logps" in inputs:
            ref_logps = inputs["old_logps"].flatten(0, 1)
        else:
            # Fallback (approximate ratio=1 if no old logps provided)
            ref_logps = per_token_logps.detach()
            
        ratio = torch.exp(per_token_logps - ref_logps)

        # 4. PREPARE ADVANTAGES WITH STIM INJECTION
        # Broadcast scalar advantage A to all tokens: (B*G, L)
        adv_per_token = advantages.unsqueeze(1).expand_as(per_token_logps).clone()

        # >>>>>> STIM INJECTION LOGIC  & METRIC LOGGING <<<<<<
        mean_rho = torch.tensor(0.0)
        if self.stim_alpha > 0:
            # We must lookup the STIM scores for these specific samples.
            # We iterate through the batch, decode the text, hash it, and find the score.
            
            # This operation is done on CPU usually, then moved to GPU
            stim_tensor = torch.zeros_like(adv_per_token, device=model.device)
            
            # Decode prompts and completions to keys
            # Note: This adds overhead but guarantees alignment without complex data collators
            prompts_text = self.processing_class.batch_decode(flat_prompts, skip_special_tokens=True)
            comps_text = self.processing_class.batch_decode(flat_completions, skip_special_tokens=True)
            
            for i, (p_txt, c_txt) in enumerate(zip(prompts_text, comps_text)):
                key = stable_hash(p_txt + c_txt)
                if key in self.stim_data:
                    scores = self.stim_data[key]
                    for pos, score in scores.items():
                        if pos < L: # Boundary check
                            stim_tensor[i, pos] = score
            
            # Modify the advantage
            # A_total(t) = A_prm + (alpha * S(t))
            adv_per_token = adv_per_token - (self.stim_alpha * stim_tensor)
            # adv_per_token = adv_per_token + stim_tensor
            # Metric Calculation
            if stim_tensor.sum() != 0:
                mean_rho = stim_tensor[stim_tensor != 0].mean().detach()

        # LOGGING DELTA TO WANDB
        # Delta = |Task_Adv - Final_Adv|
        mean_final_adv = adv_per_token.mean().detach()
        delta = (mean_final_adv - mean_task_adv).abs()
        
        if wandb.run is not None:
            wandb.log({
                "train/task_advantage": mean_task_adv,
                "train/final_advantage": mean_final_adv,
                "train/stim_delta": delta,  # <--- THIS IS THE METRIC YOU NEED
                "train/mean_stim_rho": mean_rho
            })
        # >>>>>> END STIM INJECTION <<<<<<

        # 5. GRPO Loss Calculation (PPO Clip)
        clip_eps = self.args.cliprange if hasattr(self.args, "cliprange") else 0.2
        
        part1 = ratio * adv_per_token
        part2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_per_token
        pg_loss = -torch.min(part1, part2)

        # Mask padding
        if completion_mask is not None:
            mask = completion_mask.flatten(0, 1)
            pg_loss = pg_loss * mask
            valid_tokens = mask.sum()
        else:
            valid_tokens = pg_loss.numel()

        loss = pg_loss.sum() / (valid_tokens + 1e-6)
        
        return (loss, outputs) if return_outputs else loss

# ==============================================================================
# 3. ROLLOUT FUNCTION (Connecting Offline Data to Online Trainer)
# ==============================================================================

def make_rollout_func(prompts_list, completions_list, rewards_list):
    """
    Creates a closure that TRL calls to 'generate' data. 
    Instead of generating, we feed it the data we already loaded.
    """
    # Create a lookup for fast access
    # We group by prompt to satisfy TRL's prompt-based querying
    data_map = {}
    for p, c, r in zip(prompts_list, completions_list, rewards_list):
        if p not in data_map: data_map[p] = []
        data_map[p].append((c, r))

    def rollout_func(prompts, **kwargs):
        # TRL asks for rollouts for a specific list of prompts
        batch_completions = []
        batch_rewards = []
        
        for p in prompts:
            if p in data_map:
                # We return ALL variations we have for this prompt
                # Note: TRL expects num_generations items per prompt
                items = data_map[p]
                batch_completions.extend([x[0] for x in items])
                batch_rewards.extend([x[1] for x in items])
            else:
                # Fallback if prompt mismatch (should not happen if data aligned)
                batch_completions.append("") 
                batch_rewards.append(0.0)
                
        return {
            "prompts": prompts,
            "completions": batch_completions,
            "rewards": batch_rewards
        }
    return rollout_func

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_in", type=str, required=True)
    parser.add_argument("--policy_out", type=str, required=True)
    parser.add_argument("--ref_model", type=str, required=True)
    parser.add_argument("--merged_prm", type=str, required=True)
    parser.add_argument("--stim_correct", type=str, required=True)
    parser.add_argument("--stim_wrong", type=str, required=True)
    parser.add_argument("--stim_alpha", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--kl_beta", type=float, default=0.04)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # --- INIT WANDB ---
    wandb.init(project="stim-grpo-a100", name=f"step_{args.policy_out.split('/')[-1]}")

    # 1. Load Data
    print("Loading PRM and STIM data...")
    prm_data = load_json(args.merged_prm)
    stim_map = {}
    stim_map.update(load_stim_scores(args.stim_correct))
    stim_map.update(load_stim_scores(args.stim_wrong))
    
    # Organize data for rollout_func
    # We need lists of prompts, completions, and PRM rewards
    prompts, completions, rewards = [], [], []
    
    for item in prm_data:
        p = item.get("prompt", item.get("question", item.get("input", "")))
        c = item.get("model_output", item.get("completion", item.get("response", "")))
        # Handle reward key variations
        r = item.get("reward", item.get("prm_score", item.get("final_score", 0.0)))
        
        if p and c:
            prompts.append(p)
            completions.append(c)
            rewards.append(float(r))

    unique_prompts = list(set(prompts))
    ## MAKE DATASET FOR TRL
    train_dataset = Dataset.from_dict({"prompt": unique_prompts})
    print(f"Loaded {len(prompts)} total rollouts across {len(unique_prompts)} prompts.")

    # 2. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.policy_in, trust_remote_code=True)
    tokenizer.model_input_names = ["input_ids", "attention_mask"] # Required for TRL
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- CRITICAL FIX: HANDLE NEGATIVE STEPS ---
    if args.steps == -1:
        steps_arg = -1
        epochs_arg = 1
        save_strat = "no" # Prevent crashing on negative steps
    else:
        steps_arg = args.steps
        epochs_arg = 1
        save_strat = "steps"

    # # 3. Setup Config
    # training_args = GRPOConfig(
    #     output_dir=args.policy_out,
    #     learning_rate=args.learning_rate,
    #     per_device_train_batch_size=args.batch_size,
    #     num_generations=args.num_rollouts,
    #     max_completion_length=args.max_new_tokens,
    #     num_train_epochs=1,
    #     max_steps=args.steps,
    #     save_steps=args.steps,
    #     logging_steps=1,
    #     beta=args.kl_beta,
    #     report_to="none" # Disable wandb for cleaner loop output
    # )

    training_args = GRPOConfig(
        output_dir=args.policy_out,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_rollouts,
        max_completion_length=args.max_new_tokens,
        num_train_epochs=epochs_arg,
        max_steps=steps_arg,
        save_strategy=save_strat,  # Use safe strategy
        save_steps=args.steps if args.steps > 0 else 1000, 
        logging_steps=1,
        beta=args.kl_beta,
        report_to="wandb",
        bf16=True,
        generation_kwargs={"use_cache": False}
    )

    # 4. Initialize Custom Trainer
    # We define a dummy reward_func because we pass pre-computed rewards in rollout_func
    # but TRL requires the argument.
    def dummy_reward_func(prompts, completions, **kwargs):
        return [0.0] * len(prompts)
    
    print(f"Loading model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.policy_in, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Disable cache in config
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False

    # HARD guarantee: force generate(use_cache=False) no matter what TRL passes
    _orig_generate = model.generate
    def _generate_no_cache(*a, **kw):
        kw["use_cache"] = False
        return _orig_generate(*a, **kw)
    model.generate = _generate_no_cache

    print("Disabled use_cache (config + generate monkeypatch).")

    trainer = STIMGRPOTrainer(
        stim_alpha=args.stim_alpha,
        stim_data=stim_map,     # Pass the STIM dictionary
        tokenizer=tokenizer,    # Pass tokenizer for decoding
        # model=args.policy_in,
        # model=AutoModelForCausalLM.from_pretrained(args.policy_in, trust_remote_code=True),
        model=model,
        # ref_model=args.ref_model,
        reward_funcs=dummy_reward_func, # Ignored, data overrides this
        args=training_args,
        train_dataset=train_dataset,     # We will inject dataset via rollout_func
    )

    # 5. Inject Rollout Function
    # This replaces the generation step. It feeds the loaded JSON data into the trainer.
    trainer.rollout_func = make_rollout_func(prompts, completions, rewards)

    # 6. Train
    print("Starting GRPO step (Training)...")
    # We pass the unique prompts as the "dataset" to iterate over
    # TRL will call rollout_func(unique_prompts[batch])
    trainer.train()

    # 7. Save
    print(f"Saving model to {args.policy_out}...")
    trainer.save_model(args.policy_out)
    tokenizer.save_pretrained(args.policy_out)

if __name__ == "__main__":
    main()