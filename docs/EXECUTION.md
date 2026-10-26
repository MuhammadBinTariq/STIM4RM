# Git, Colab execution and monitoring

This guide uses the implemented CLI. The default scorer remains a retokenized,
bounded, partial-corpus STIM proxy. GPU speed and live API coverage are unmeasured.

## 1. Commit and publish

Run in PowerShell from the local repository:

```powershell
git switch refactor/v2
git status --short
git diff --check
git diff -- src/gen_eval.sh src/mem_cal.sh src/token_select_alter.sh
git add -- .gitignore README.md requirements-v2.txt configs docs notebooks scripts
git add -- src/__init__.py src/train.py src/evaluate.py src/prepare.py src/rewardmaxxing tests/smoke_test.py
# These legacy files currently have zero substantive additions/deletions.
# Inspect the diff above before refreshing their line-ending-normalized index entries.
git add -- src/gen_eval.sh src/mem_cal.sh src/token_select_alter.sh
git diff --cached --stat
git diff --cached --check
git diff --cached
git ls-files --others --exclude-standard
& .venv-v2/Scripts/python.exe tests/smoke_test.py
# Continue only after successful tests.
git commit -m "Implement reproducible RewardMaxxing v2 training and Colab experiments"
git fetch origin
git log --oneline --graph --decorate -8
git status --short --untracked-files=all
git tag -a v2.0.0-rc1 -m "CPU-verified RewardMaxxing v2 experiment runner"
git push -u origin refactor/v2
git push origin v2.0.0-rc1
git rev-parse HEAD
```

This adds commits to refactor/v2, preserving legacy ancestors and leaving main
unchanged. Do not force-push. If the remote branch rejects the push, inspect
and deliberately merge its changes, rerun tests, then tag the resulting commit.
Use a fresh tag name if v2.0.0-rc1 already exists; never move a published tag.
Run commands in order and stop on an error.

After committing, status and the untracked-file listing should be empty.
Virtualenvs, .tmp, runs, SQLite, dist, evaluation-data, .env secrets and W&B files
are intentionally ignored; do not delete them with git clean. Results belong on
Drive. The three legacy scripts have no substantive diff in the current tree.

Open the pushed notebook:
https://colab.research.google.com/github/MuhammadBinTariq/STIM4RM/blob/refactor/v2/notebooks/RewardMaxxing_v2.ipynb

The notebook clones refactor/v2 through HTTPS, then detaches at a fixed commit.
Paste the full SHA into COMMIT before formal experiments. It refuses a dirty
existing checkout. Source SHA, dirty status, environment and arguments are saved
under each run's invocations directory.

```bash
git clone --branch refactor/v2 --single-branch https://github.com/MuhammadBinTariq/STIM4RM.git /content/rewardmaxxing-v2
cd /content/rewardmaxxing-v2
git checkout --detach YOUR_FULL_COMMIT_SHA
```

To update a normal branch checkout *between experiments*, use git switch
refactor/v2 followed by git pull --ff-only origin refactor/v2. For the notebook's
detached checkout, fetch then check out a new reviewed SHA. Never update source
halfway through a paired sweep; use a new output root for a new source revision.

## 2. Hardware and the 800-CU budget

Start with A100 for diagnostic headroom. Test L4 with the identical recipe in a
separate pilot root, then choose by measured cost per completed update:

```text
CU/update = current displayed CU/hour * seconds/update / 3600
training CU = seeds * updates * (baseline_seconds + treatment_seconds) / 3600 * CU/hour
total CU = training CU + measured evaluation CU + startup/retry allowance
```

Read the actual CU/hour in your Colab Resources panel. Historical prices and
Cloud/Enterprise hourly prices are not a reliable conversion for your account.
The runner requires BF16 and >=20 GiB VRAM. L4/A100 are intended candidates;
other architectures need compatibility checks with pinned CUDA/PyTorch/
bitsandbytes. Use the same GPU type for paired arms.

Suggested envelopes: 40 CU systems pilots, 80 CU development selection, 360 CU
primary training, 160 CU evaluation, 160 CU reserve. Reallocate after measurements;
these are not runtime forecasts. The default 512 prompts / 2 prompts per update
means 256 updates per epoch and 1,536 updates for two arms and three seeds.
If HTTP dominates, a faster GPU may not improve CU/update.

Colab GPU availability varies. Pro+ supports up to 24 hours of continuous
execution with sufficient compute units; checkpoint/resume remains necessary.
Disconnect/delete the runtime after work. Official limits:
https://research.google.com/colaboratory/faq.html

## 3. First 25-update systems pilot

The notebook uses uv to install Python 3.12 into /content/rewardmaxxing-venv-py312,
independently of the system/kernel Python version, and runs the CPU smoke test,
and keeps HF_HOME and live SQLite on local SSD. Only snapshots/results go to Drive.
The following Bash variables are reused below; the notebook has equivalent
subprocess.run calls with explicit working directories.

```bash
cd /content/rewardmaxxing-v2
PY=/content/rewardmaxxing-venv-py312/bin/python
OUT=/content/drive/MyDrive/RewardMaxxing-v2
RESOLVED="$OUT/resolved.json"
# Resolve once and reuse. This refuses to overwrite an existing file.
"$PY" -m src.prepare --config configs/rewardmaxxing.json --output "$RESOLVED"

for arm in baseline rewardmaxxing; do
  "$PY" scripts/colab_run.py --resolved-config "$RESOLVED" \
    --output-root "$OUT/pilot-a100-lambda01" --arm "$arm" --phase pilot \
    --seed 42 --pilot-steps 25 --stim-lambda 0.1 --monitor-stim
done
```

Baseline always forces lambda=0; monitor-stim measures without changing its loss.
Use pilot-l4-lambda01 for a separate L4 test.

| Parameter | Shared setting |
|---|---|
| Model | allenai/OLMo-2-0425-1B-Instruct, pinned revision |
| Prompts / rollouts per prompt | 2 / 4, i.e. 8 responses/update |
| Generation / update microbatch | 2 / 1 response |
| Response / prompt caps | 256 / 768 tokens |
| Policy / LoRA | BF16; rank 16, alpha 32, dropout 0 |
| AdamW | learning rate 2e-5, weight decay 0, grad norm 1 |
| KL / clipping / advantage floor | 0.04 / 0.2 / 0.05 |
| Sampling | temperature 1, top-p 1, top-k disabled |
| Reward | frozen NF4 VersaPRM, mean step score |
| STIM | additive; K=8; <=4 positions; <=32 context tokens |
| Gate / count support | probability >=0.3; count >=5; >=3 distinct counts |
| Save / retain | every 20 plus stop boundary / latest 2 checkpoints |
| Train / dev | 512 / 128; fixed split seed 1729 |

Change these settings in a copied resolved JSON before starting a new recipe,
not through nonexistent trainer flags. --train-steps is a cumulative stop limit.
Changing training semantics refuses resume. A different lambda needs a separate root.
Both arms start from the same pinned base and same-seed LoRA initialization;
zero-B LoRA initially leaves the base function unchanged. Prompt order and seeds
match, but responses naturally diverge after the policies update.

## 4. Pilot telemetry and local logging

- metrics/step-NNNNNN.json: rewards, exact match, STIM/coverage, KL, gradients,
  truncation, cache/request statistics, phase timing and CUDA memory.
- system.jsonl: nvidia-smi utilization and used GPU memory every 5 seconds;
  checkpoint/run/cache size and local disk free space every 60 seconds.
- storage.json: final adapter and optimizer sizes and unexpected full-model files.
- pilot-summary.json: aggregate timings and cache hit rate.
- invocations/: source, hardware, environment and command provenance.

Local/Drive logging is the default; no W&B dependency, credentials or uploads.

```bash
RUN="$OUT/pilot-a100-lambda01/rewardmaxxing-seed42"
"$PY" scripts/export_metrics.py --run-dir "$RUN" --output "$RUN/metrics.csv"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
df -h /content
du -sh /content/huggingface /content/rewardmaxxing-cache "$RUN"
```

Plot step against stim_rho, stim_positive, stim_valid_fraction, task_reward,
correct and kl. Also inspect cache_hit_rate, request_mean_seconds,
stim_wait_seconds, persistence_seconds, cuda_allocated_gb and cuda_reserved_gb.
Missing STIM is null, not zero. Use monitor-stim in both arms for comparable
training telemetry; independent held-out audits remain the primary STIM measure,
because training gates depend on the current policy.

Cache hit rate = durable hits / (hits + misses + shared in-flight lookups).
Request time includes retries/backoff per miss; summed concurrent request time
can exceed wall time. Report cold and warm phases separately; shared-cache reuse
between arms is useful but is not evidence of an architectural speedup.

Inspect the first 5 steps as warm-up, then the following 20. Suggested engineering
targets, not established scientific thresholds:

- Zero HTTP failures/nonfinite gradients, and meaningful STIM coverage. Start
  with >=50% valid selected positions as a pilot target; inspect invalid statuses
  rather than treating missing values as zero or blindly weakening thresholds.
- Prefer exposed STIM wait <20% of wall time. Above 50%, investigate HTTP before
  spending on more GPU. Public API performance is not guaranteed.
- Keep about 20% VRAM headroom. Allocated memory should settle at comparable
  sequence lengths after optimizer initialization. Reserved memory can stay high
  due to allocator caching; nvidia-smi includes allocations outside PyTorch.
- After saves, expect <=2 checkpoints (temporarily 3 during atomic save/retention).
  Each contains adapters AND LoRA optimizer/RNG state, not a full base model.
- Inspect truncation/format failures. If 256 tokens is insufficient, raise the cap
  consistently in a new paired recipe rather than only one arm.
- Keep >=20 GiB local disk headroom. NF4 reduces resident weights, not original
  Hub downloads. Logs/cache still grow. Check Drive quota separately.

Telemetry observes rather than auto-killing jobs. STIM failures stop incomplete
treatment batches and preserve resumable state. Do not call empty_cache every
step as a substitute for diagnosing live tensors.

## 5. Primary sweep and lambda selection

Use lambda=0.1 as the primary treatment. Test 0.2 on DEVELOPMENT data before
deciding on an ablation. Never use GSM8K test or public twins to choose lambda.

```bash
for seed in 42 43 44; do
  for arm in baseline rewardmaxxing; do
    "$PY" scripts/colab_run.py --resolved-config "$RESOLVED" \
      --output-root "$OUT/primary-lambda01" --arm "$arm" --phase train \
      --seed "$seed" --train-steps 256 --stim-lambda 0.1 --monitor-stim
  done
done
```

Six sequential processes avoid concurrent model copies. Repeat a command to resume.
Use a fresh primary root after tuning. To test lambda=0.2, use a separate root
such as dev-lambda02 and --stim-lambda 0.2, with a matched baseline and equal
update budget. Do not warm-start it from a trained lambda=0.1 adapter.

## 6. Twins and low-memory evaluation

The helper downloads official GSM-Symbolic main (numeric) and p1 (extra structure),
one fixed instance per parent, pins a dataset SHA and records file hashes. It
preserves references and categories; no guessed labels or arbitrary number swaps.

```bash
"$PY" scripts/prepare_twins.py --output-dir "$OUT/evaluation-data/symbolic-one" \
  --subsets main p1 --instances 1
RUN="$OUT/primary-lambda01/rewardmaxxing-seed42"
CKPT="$RUN/step-000256"
TWINS="$OUT/evaluation-data/symbolic-one"

# Process 1: policy + KL + STIM, no PRM.
"$PY" -m src.evaluate run --checkpoint "$CKPT" \
  --output "$RUN/paired-generations.json" --skip-prm --stim-audit \
  --audit-positions 8 --max-new-tokens 512 --paired-only \
  --twins "$TWINS/main.jsonl" "$TWINS/p1.jsonl"

# Process 2: PRM only; original responses and KL remain unchanged.
"$PY" -m src.evaluate score-prm --input "$RUN/paired-generations.json" \
  --output "$RUN/paired-scored.json" --config "$RUN/config.json"

# Separate headline accuracy on ALL 1,319 GSM8K test questions.
"$PY" -m src.evaluate run --checkpoint "$CKPT" \
  --output "$RUN/gsm8k-full-generations.json" --skip-prm --max-new-tokens 512
```

Repeat for both arms and all seeds. Paired-only GSM8K accuracy is on the twins'
parent subset, not the full test set. Removing paired-only evaluates all originals
plus supplied twins, with a correspondingly larger audit API cost. Report parent
and instance counts; never compare runs with different evaluation panels.

```bash
P="$OUT/primary-lambda01"
"$PY" -m src.evaluate compare \
  --baseline "$P/baseline-seed42/paired-scored.json" "$P/baseline-seed43/paired-scored.json" "$P/baseline-seed44/paired-scored.json" \
  --rewardmaxxing "$P/rewardmaxxing-seed42/paired-scored.json" "$P/rewardmaxxing-seed43/paired-scored.json" "$P/rewardmaxxing-seed44/paired-scored.json" \
  --output "$P/paired-comparison.json"
```

For GSM-Plus, the official dataset is qintongli/GSM-Plus. Pin the SHA using
HfApi.dataset_info, load its test split at that revision, and export JSONL.
The evaluator accepts question, answer, seed_question and perturbation_type and
verifies parents against GSM8K test. Unparseable numeric references are rejected,
not silently relabeled. Full GSM-Plus is much larger than the initial panel.

Primary dataset documentation:
- https://huggingface.co/datasets/apple/GSM-Symbolic
- https://huggingface.co/datasets/qintongli/GSM-Plus

## 7. Proxy versus gold checkpoint frontier

Use a fixed DEVELOPMENT panel at cumulative steps 0,64,128,256. Stop at each
limit, generate/evaluate, then resume. Retention=2 will otherwise delete older
adapters. Saved generations remain PRM-scoreable using run/config.json even after
the originating adapter is deleted. Initial evaluation can use --config before
training; keep the same pinned model, panel and generation protocol.

```bash
# After stopping at step 64:
"$PY" -m src.evaluate run --checkpoint "$RUN/step-000064" \
  --split dev --limit 64 --output "$RUN/dev64-generations.json" \
  --skip-prm --stim-audit --max-new-tokens 512
"$PY" -m src.evaluate score-prm --input "$RUN/dev64-generations.json" \
  --output "$RUN/dev64-scored.json" --config "$RUN/config.json"
# Repeat at 128 and 256, then collect:
"$PY" -m src.evaluate frontier --inputs "$RUN/dev64-scored.json" \
  "$RUN/dev128-scored.json" "$RUN/dev256-scored.json" \
  --output "$RUN/development-frontier.json"
```

Plot mean PRM reward on x, accuracy on y, with update/lambda labels; inspect KL,
high-PRM wrong rate, length and truncation. Points are descriptive, not proof of
reward hacking. Use matched seeds and bootstrap comparisons for uncertainty.
Do not select checkpoints on test outcomes.

score-prm loads only VersaPRM, scores one response at a time, never regenerates,
and resumes by input hash. A small identical file can be rescored with
--unquantized on A100 for PRM calibration if memory permits; that is a different
scoring protocol. For generation OOM, use a smaller head chunk in a new matched
recipe. For PRM-only OOM, use more VRAM. Do not silently truncate or substitute
rewards in only one arm.
