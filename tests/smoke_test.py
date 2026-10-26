"""Offline CPU integration tests: real tiny OLMo2 + PEFT, no pretrained weights.

Run from any directory: python tests/smoke_test.py
"""
import os
import sys
import time
import faulthandler

print(f"[smoke] Starting Python {sys.version.split()[0]} (offline CPU tests)", flush=True)
faulthandler.enable()
# Show the exact stack if imports or a test stall; never terminate the process.
faulthandler.dump_traceback_later(60, repeat=True)
startup = time.perf_counter()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / ".tmp").mkdir(exist_ok=True)
print("[smoke] Importing PyTorch...", flush=True)
import torch
print(f"[smoke] PyTorch {torch.__version__} loaded; importing PEFT/Transformers...", flush=True)
import peft
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast

from src.rewardmaxxing.cache import InfiniGramClient, QueryFailure, SQLiteCache
from src.rewardmaxxing.checkpoints import load_training_state, save_checkpoint, seed_all
from src.rewardmaxxing.config import StimConfig, TrainConfig
from src.rewardmaxxing.data import validate_splits
from src.rewardmaxxing.objective import forward_kl, group_advantages, shape_advantages, token_loss
from src.rewardmaxxing.policy import Policy, Rollout, tokenizer_fingerprint
from src.rewardmaxxing.rewards import exact_match, parse_answer
from src.rewardmaxxing.stim import StimScorer, select_positions, spearman
from src.rewardmaxxing.trainer import train, update
from src.rewardmaxxing.metrics import TrainingOverlap, lcs_length, paired_comparison

torch.set_num_threads(1)
print(f"[smoke] Dependencies ready in {time.perf_counter()-startup:.1f}s; running tests...", flush=True)


def tiny_tokenizer():
    tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "A", "B", "1", "2", "3", "4", "+", "=",
              "remaining", "####"] + [f"word{i}" for i in range(50)]
    raw = Tokenizer(WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", bos_token="[BOS]",
                                 eos_token="[EOS]", unk_token="[UNK]")
    tok.padding_side = "left"
    return tok


def tiny_base():
    seed_all(77)
    config = Olmo2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                        max_position_embeddings=128, pad_token_id=0, bos_token_id=1,
                        eos_token_id=2, attention_dropout=0.0)
    return Olmo2ForCausalLM(config)


def tiny_policy(adapter=None):
    base = tiny_base()
    if adapter:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return Policy(model, tiny_tokenizer(), head_chunk_size=2)


def fixtures():
    return [Rollout(f"sample-{i}", i//2, "A + B", "#### 3", [1, 4, 5], ids,
                    "1 + 2 = 3\n#### 3", reward=[0.9, 0.2, 0.3, 0.7][i])
            for i, ids in enumerate([[6,10,7,11,8,2], [9,10,8,2], [6,7,8,2], [7,6,10,8,2]])]


def config(weight=0.2):
    return TrainConfig(group_size=2, beta=0.04, stim_lambda=weight, learning_rate=0.01,
                       head_chunk_size=2, stim=StimConfig(acknowledge_proxy=True))


class SmokeTests(unittest.TestCase):
    def test_atomic_replace_retries_only_windows_locks(self):
        from unittest.mock import patch
        from src.rewardmaxxing.checkpoints import atomic_replace
        locked = PermissionError("simulated Windows directory lock")
        locked.winerror = 5
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            source, destination = Path(directory)/"staging", Path(directory)/"published"
            source.mkdir()
            (source/"payload").write_text("complete")
            real_replace = os.replace
            calls = []
            def transient(a, b):
                calls.append((a,b))
                if len(calls) < 3:
                    raise locked
                return real_replace(a,b)
            with patch("src.rewardmaxxing.checkpoints.os.replace", side_effect=transient), \
                 patch("src.rewardmaxxing.checkpoints.time.sleep") as sleep:
                atomic_replace(source,destination)
            self.assertEqual(len(calls),3)
            self.assertEqual(sleep.call_count,2)
            self.assertEqual((destination/"payload").read_text(),"complete")
            with patch("src.rewardmaxxing.checkpoints.os.replace", side_effect=locked) as replace, \
                 patch("src.rewardmaxxing.checkpoints.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    atomic_replace(source,destination)
                self.assertEqual(replace.call_count,8)
                self.assertEqual(sleep.call_count,7)
            with patch("src.rewardmaxxing.checkpoints.os.replace",
                       side_effect=PermissionError("ordinary permission failure")) as replace:
                with self.assertRaises(PermissionError):
                    atomic_replace(source,destination)
                replace.assert_called_once()

    def test_failed_checkpoint_publication_keeps_latest_unchanged(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            policy, cfg = tiny_policy(), config()
            optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad])
            root = Path(directory)
            sentinel = '{"checkpoint":"previous","step":0}'
            (root/"latest.json").write_text(sentinel)
            real_replace = os.replace
            locked = PermissionError("simulated persistent lock")
            locked.winerror = 32
            def fail_directory(a,b):
                if Path(a).is_dir():
                    raise locked
                return real_replace(a,b)
            with patch("src.rewardmaxxing.checkpoints.os.replace", side_effect=fail_directory), \
                 patch("src.rewardmaxxing.checkpoints.time.sleep"):
                with self.assertRaises(PermissionError):
                    save_checkpoint(root,policy,optimizer,cfg,1,{"fixture":True})
            self.assertEqual((root/"latest.json").read_text(),sentinel)
            self.assertFalse((root/"step-000001").exists())
            staged = list(root.glob(".step-000001-*.part"))
            self.assertEqual(len(staged),1)
            self.assertTrue((staged[0]/"training_state.pt").is_file())

    def test_generation_slicing_and_teacher_forcing(self):
        policy = tiny_policy()
        cfg = config(); cfg.max_prompt_tokens = 60; cfg.max_new_tokens = 5
        cfg.generation_batch_size = 3
        examples = [{"id":"short", "question":"A", "answer":"#### 3"},
                    {"id":"long", "question":"A + B = 3", "answer":"#### 3"}]
        records = policy.generate(examples, cfg)
        self.assertEqual(len(records), 4)
        self.assertEqual([r.group_id for r in records], [0,0,1,1])
        policy.score_rollouts(records, 8)
        for r in records:
            self.assertLessEqual(len(r.completion_ids), 5)
            ids = torch.tensor([r.prompt_ids + r.completion_ids])
            with torch.no_grad():
                logits = policy.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]
                start = len(r.prompt_ids)-1
                lp = logits[start:start+len(r.completion_ids)].float().log_softmax(-1)
                expected = lp.gather(-1, torch.tensor(r.completion_ids)[:,None]).squeeze(-1)
            torch.testing.assert_close(r.old_log_probs, expected, atol=1e-6, rtol=1e-5)

    def test_advantage_signs_and_zero_variance(self):
        a = group_advantages(torch.tensor([0.2, 0.8, 0.5, 0.5]), torch.tensor([0,0,1,1]))
        torch.testing.assert_close(a, torch.tensor([-1., 1., 0., 0.]))
        p = torch.tensor([[0.8], [0.8], [0.], [0.]])
        for kind in ["additive", "multiplicative"]:
            shaped = shape_advantages(a, p, 0.2, kind)
            self.assertLess(shaped[0,0], a[0])
            self.assertLess(shaped[1,0], a[1])
        with self.assertRaises(ValueError):
            shape_advantages(a, -p, 0.2)

    def test_exact_kl_and_action_gradient(self):
        logits = torch.tensor([[0.4, -0.2, 0.8]], requires_grad=True)
        reference = torch.tensor([[0.1, 0.3, -0.1]], requires_grad=True)
        expected = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=logits),
                                                   torch.distributions.Categorical(logits=reference))
        torch.testing.assert_close(forward_kl(logits, reference), expected)
        forward_kl(logits, reference).sum().backward()
        self.assertIsNone(reference.grad)
        gradients = []
        for advantage in [1.0, 0.6]:
            z = logits.detach().clone().requires_grad_()
            old = z.log_softmax(-1)[:,0].detach()
            loss = token_loss(z, reference, torch.tensor([0]), old,
                              torch.tensor([advantage]), 0.2, 0.0)[0].sum()
            loss.backward()
            gradients.append(z.grad[0,0].item())
        self.assertGreater(gradients[1], gradients[0])  # Smaller encouragement of penalized action.

    def test_real_lora_update_and_frozen_reference(self):
        policy = tiny_policy()
        records = fixtures()
        policy.score_rollouts(records, 8)
        for record in records:
            n = len(record.completion_ids)
            self.assertEqual(record.old_log_probs.shape, (n,))
            self.assertEqual(record.top_ids.shape, (n,8))
            self.assertFalse(record.old_log_probs.requires_grad)
            torch.testing.assert_close(policy.kl_for_record(record), torch.zeros(n), atol=1e-6, rtol=0)
        baseline = Policy(copy.deepcopy(policy.model), tiny_tokenizer(), 2)
        before = {k: v.detach().clone() for k,v in policy.model.named_parameters()}
        with policy.model.disable_adapter(), torch.no_grad():
            reference_before = policy._hidden(records[0]).clone()
        opt = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=0.01)
        bopt = torch.optim.AdamW([p for p in baseline.model.parameters() if p.requires_grad], lr=0.01)
        penalty = torch.zeros((4,6)); penalty[:,0] = 0.8
        # Padding values must never enter the loss.
        padded = penalty.clone()
        for i,r in enumerate(records):
            padded[i,len(r.completion_ids):] = 1
        stats = update(policy, records, padded, opt, config())
        bstats = update(baseline, records, penalty, bopt, config(0.0))
        self.assertTrue(stats["gradient_norm"] > 0)
        self.assertAlmostEqual(stats["ratio"], 1.0, places=5)
        self.assertTrue(abs(stats["loss"]-bstats["loss"]) > 0.01)
        changed = []
        for name, value in policy.model.named_parameters():
            if not value.requires_grad:
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
                self.assertIsNone(value.grad)
            elif not torch.equal(value, before[name]):
                changed.append(name)
        self.assertTrue(changed)
        self.assertTrue(all("lora_" in name for name in changed))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in policy.model.parameters() if p.requires_grad))
        policy.model.eval()
        with policy.model.disable_adapter(), torch.no_grad():
            torch.testing.assert_close(policy._hidden(records[0]), reference_before, atol=0, rtol=0)
        self.assertGreater(policy.kl_for_record(records[0]).mean().item(), 0)

    def test_chunked_loss_matches_dense_gradient(self):
        policy = tiny_policy()
        record = fixtures()[0]
        policy.score_rollouts([record], 8)
        advantage = torch.linspace(-1, 1, len(record.completion_ids))
        loss, _ = policy.loss_for_record(record, advantage, 0.2, 0.04, len(advantage))
        loss.backward()
        gradients = {n:p.grad.clone() for n,p in policy.model.named_parameters() if p.grad is not None}
        policy.model.zero_grad(set_to_none=True)
        policy.model.eval()
        with policy.model.disable_adapter(), torch.no_grad():
            ref = policy.head(policy._hidden(record))
        policy.model.train()
        logits = policy.head(policy._hidden(record))
        dense = token_loss(logits, ref, torch.tensor(record.completion_ids), record.old_log_probs,
                           advantage, 0.2, 0.04)[0].mean()
        dense.backward()
        torch.testing.assert_close(loss, dense)
        for n,p in policy.model.named_parameters():
            if n in gradients:
                torch.testing.assert_close(p.grad, gradients[n], atol=2e-6, rtol=1e-4)

    def test_checkpoint_resume_matches_next_update(self):
        policy, cfg = tiny_policy(), config()
        optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=0.01)
        records = fixtures(); policy.score_rollouts(records, 8)
        penalty = torch.zeros(4,6); penalty[:,1] = 0.5
        update(policy, records, penalty, optimizer, cfg)
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            path = save_checkpoint(directory, policy, optimizer, cfg, 1, {"fixture": True})
            expected_random = torch.rand(4)
            next_records = fixtures(); policy.score_rollouts(next_records, 8)
            update(policy, next_records, penalty, optimizer, cfg)
            resumed = tiny_policy(path/"adapter")
            ropt = torch.optim.AdamW([p for p in resumed.model.parameters() if p.requires_grad], lr=0.01)
            self.assertEqual(load_training_state(path, ropt, cfg), 1)
            torch.testing.assert_close(torch.rand(4), expected_random)
            resumed_records = fixtures(); resumed.score_rollouts(resumed_records, 8)
            update(resumed, resumed_records, penalty, ropt, cfg)
            for (n,p),(rn,rp) in zip(policy.model.named_parameters(), resumed.model.named_parameters()):
                self.assertEqual(n,rn)
                torch.testing.assert_close(p, rp, atol=1e-7, rtol=1e-6)

    def test_cache_persistence_singleflight_backup_and_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            path = Path(directory)/"cache.sqlite"
            calls = []
            def transport(payload):
                calls.append(payload)
                time.sleep(0.03)
                return {"count": 0, "approx": False}
            client = InfiniGramClient(SQLiteCache(path), "fixture", "tok-v1", transport)
            with ThreadPoolExecutor(max_workers=8) as executor:
                values = list(executor.map(lambda _: client.count([1,2]), range(8)))
            self.assertEqual(values, [0]*8)
            self.assertEqual(len(calls), 1)
            client.cache.snapshot(Path(directory)/"drive.sqlite")
            client.close()
            def forbidden(_):
                raise AssertionError("Persistent hit must not use network")
            restored = InfiniGramClient(SQLiteCache(path), "fixture", "tok-v1", forbidden)
            self.assertEqual(restored.count([1,2]), 0)
            restored.close()
            copied = SQLiteCache(Path(directory)/"fresh.sqlite")
            self.assertTrue(copied.restore_if_empty(Path(directory)/"drive.sqlite"))
            verified = InfiniGramClient(copied, "fixture", "tok-v1", forbidden)
            self.assertEqual(verified.count([1,2]), 0)
            verified.close()
            bad = InfiniGramClient(SQLiteCache(path), "fixture", "tok-v2",
                                   lambda _: {"error": "temporary failure"})
            with self.assertRaises(QueryFailure):
                bad.count([1,2])
            bad.transport = lambda _: {"count": 7, "approx": False}
            self.assertEqual(bad.count([1,2]), 7)
            self.assertEqual(bad.stats["misses"], 2)
            bad.close()

    def test_stim_alignment_gating_ties_and_negative_rho(self):
        tokenizer = tiny_tokenizer()
        cfg = StimConfig(mode="native", top_k=3, max_positions=2, min_probability=0.0,
                         min_support=5, min_distinct_counts=3)
        record = Rollout("fixture", 0, "Q", "#### 3", [1], [4,6,7,8,2], "A 1 2 3")
        record.old_log_probs = torch.full((5,), -0.5)
        record.top_ids = torch.tensor([[6,7,8]]*5)
        record.top_probs = torch.tensor([[0.5,0.3,0.2]]*5)
        self.assertEqual(len(select_positions(record, tokenizer, cfg)), 2)
        self.assertEqual(spearman([3,2,1],[1,2,3]), -1)
        self.assertIsNone(spearman([3,2,1],[0,0,0]))
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            counts = {(6,):100, (4,):20, (4,6):9, (4,7):5, (4,8):1}
            client = InfiniGramClient(SQLiteCache(Path(directory)/"c.sqlite"), "fixture",
                tokenizer_fingerprint(tokenizer), lambda p: {"count":counts.get(tuple(p["query_ids"]),0), "approx":False})
            scorer = StimScorer(cfg, tokenizer, tokenizer, client)
            try:
                penalty, observations = scorer.score([record], positions=[[1]])
                self.assertEqual(penalty.shape, (1,5))
                self.assertEqual(penalty[0,1], 1)
                self.assertEqual(penalty.sum(), 1)
                self.assertEqual(observations[0].status, "ok")
                record.top_probs[:] = torch.tensor([0.2,0.3,0.5])
                penalty, observations = scorer.score([record], positions=[[1]])
                self.assertEqual(observations[0].rho, -1)
                self.assertEqual(penalty.sum(), 0)
            finally:
                scorer.close()

    def test_stim_failure_preserves_api_cause_and_position(self):
        tokenizer = tiny_tokenizer()
        cfg = StimConfig(mode="native", top_k=3, max_positions=2, min_probability=0.0)
        record = Rollout("api-fixture",0,"Q","#### 3",[1],[4,6,7],"A 1 2")
        record.old_log_probs = torch.full((3,), -0.5)
        record.top_ids = torch.tensor([[6,7,8]]*3)
        record.top_probs = torch.tensor([[0.5,0.3,0.2]]*3)
        def fail(_):
            raise QueryFailure("HTTP 429 fixture: retry later")
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            client = InfiniGramClient(SQLiteCache(Path(directory)/"c.sqlite"),"fixture","v1",fail)
            scorer = StimScorer(cfg,tokenizer,tokenizer,client)
            try:
                with self.assertRaises(QueryFailure) as caught:
                    scorer.score([record],positions=[[1]])
                self.assertIn("1/1 selected token positions",str(caught.exception))
                self.assertIn("HTTP 429 fixture",str(caught.exception))
                observation = caught.exception.observations[0]
                self.assertEqual(observation["position"],1)
                self.assertIsNone(observation["rho"])
                self.assertIn("HTTP 429 fixture",observation["error"])
            finally:
                scorer.close()

    def test_answers_and_split_isolation(self):
        self.assertEqual(str(parse_answer("#### -5")), "-5")
        self.assertEqual(str(parse_answer("#### 3/4")), "3/4")
        self.assertTrue(exact_match("#### 0.75", "#### 3/4"))
        self.assertTrue(exact_match("#### 1,000.", "1000"))
        self.assertIsNone(parse_answer("Work contains 5 but no final answer"))
        self.assertIsNone(parse_answer("#### 1/0"))
        with self.assertRaises(ValueError):
            validate_splits([{"question":"same"}], [], [{"question":"same"}])
        with self.assertRaises(ValueError):
            TrainConfig().validate()

    def test_overlap_and_paired_evaluation(self):
        overlap = TrainingOverlap(["A B C D E F G H", "some other solution"])
        values = overlap.score("A B C D E F G H")
        self.assertEqual(values["max_train_rouge_l_f1"], 1)
        self.assertEqual(values["train_4gram_coverage"], 1)
        self.assertEqual(values["train_8gram_coverage"], 1)
        # LCS bit-parallel implementation versus scalar dynamic programming.
        import random
        random.seed(9)
        for _ in range(20):
            a, b = ([random.randrange(4) for _ in range(n)] for n in (11,8))
            masks = {}
            for i,t in enumerate(b): masks[t] = masks.get(t,0) | 1 << i
            row = [0]*(len(b)+1)
            for x in a:
                previous, row = row, [0]*(len(b)+1)
                for j,y in enumerate(b,1):
                    row[j] = previous[j-1]+1 if x==y else max(previous[j],row[j-1])
            self.assertEqual(lcs_length(a,masks), row[-1])
        metadata = {k:"same" for k in ("seed","model_id","model_revision","dataset_revision",
                    "eval_max_new_tokens","training_manifest_hash","evaluation_input_hash","reward","stim_audit",
                    "split","decoding","metric_tokenizer","prm_scored","prm_quantized",
                    "prm_id","prm_revision","prm_base_revision","training_protocol","checkpoint_name")}
        rows = [{"id":str(i), "original_id":str(i//2), "split":"gsm8k", "correct":0,
                 "question":str(i), "reference":"#### 1"} for i in range(6)]
        base = {"metadata":{**metadata,"lambda":0}, "rows":rows}
        treatment = {"metadata":{**metadata,"lambda":0.2}, "rows":[{**r,"correct":1} for r in rows]}
        result = paired_comparison([base],[treatment], bootstrap=100)
        self.assertEqual(result["gsm8k/correct"]["delta"], 1)
        self.assertEqual(result["gsm8k/correct"]["ci95"], [1,1])
        with self.assertRaises(ValueError):
            paired_comparison([base,base], [treatment,treatment], bootstrap=100)
        changed = copy.deepcopy(treatment)
        changed["metadata"]["prm_revision"] = "different"
        with self.assertRaises(ValueError):
            paired_comparison([base], [changed], bootstrap=100)

    def test_failed_scoring_resumes_pending_without_regeneration(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.rewardmaxxing.stim import Observation
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            policy, cfg = tiny_policy(), config()
            cfg.output_dir = directory
            cfg.max_steps = 1
            policy.generate = Mock(return_value=fixtures())
            reward = SimpleNamespace(score=Mock())
            scorer = SimpleNamespace(client=SimpleNamespace(stats={}),
                score=Mock(side_effect=QueryFailure("simulated HTTP outage")))
            optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=0.01)
            with self.assertRaises(QueryFailure):
                train(policy, reward, scorer, optimizer, cfg, [[]], {"fixture":True})
            self.assertTrue((Path(directory)/"pending.pt").exists())
            failures = list((Path(directory)/"failures").glob("*.json"))
            self.assertEqual(len(failures),1)
            failure = json.loads(failures[0].read_text())
            self.assertEqual(failure["error"],"simulated HTTP outage")
            self.assertTrue(failure["pending_saved"])
            self.assertFalse(optimizer.state)
            initial = Path(directory)/"step-000000"
            resumed = tiny_policy(initial/"adapter")
            resumed.generate = Mock(side_effect=AssertionError("Must reuse saved rollout"))
            resumed_optimizer = torch.optim.AdamW([p for p in resumed.model.parameters() if p.requires_grad], lr=0.01)
            start = load_training_state(initial, resumed_optimizer, cfg)
            observations = [Observation("sample-0", 0, "ok", rho=0.5)]
            scorer.score = Mock(return_value=(torch.zeros(4,6), observations))
            train(resumed, reward, scorer, resumed_optimizer, cfg, [[]], {"fixture":True}, start)
            resumed.generate.assert_not_called()
            self.assertEqual(reward.score.call_count, 1)
            self.assertFalse((Path(directory)/"pending.pt").exists())
            self.assertTrue((Path(directory)/"step-000001"/"training_state.pt").exists())

    def test_posthoc_prm_resume_preserves_generations_and_kl(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from src.rewardmaxxing.posthoc import score_prm_file, frontier
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            root = Path(directory)
            cfg = config()
            cfg.prm_revision, cfg.prm_base_revision = "pinned-adapter", "pinned-base"
            (root/"config.json").write_text(json.dumps(cfg.to_dict()))
            metadata = {"prm_id":cfg.prm_id, "prm_revision":cfg.prm_revision,
                        "prm_base_revision":cfg.prm_base_revision, "prm_scored":False,
                        "seed":42, "evaluation_input_hash":"panel", "eval_max_new_tokens":512,
                        "split":"dev", "decoding":"greedy", "training_protocol":"recipe",
                        "checkpoint_name":"step-000064", "lambda":0.1}
            rows = [{"id":str(i), "original_id":str(i), "split":"gsm8k", "question":"Q",
                "reference":"#### 3", "response":"Work\n#### 3", "truncated":False,
                "correct":1, "kl":0.125, "prm_reward":None} for i in range(2)]
            source, output = root/"generated.json", root/"scored.json"
            source.write_text(json.dumps({"metadata":metadata,"rows":rows}))
            original_bytes = source.read_bytes()
            calls = []
            def score(records):
                calls.append(records[0].sample_id)
                if calls == ["0", "1"]:
                    raise RuntimeError("simulated interruption")
                records[0].reward, records[0].step_scores = 0.8, [0.8]
            factory = lambda cfg, device:SimpleNamespace(score=score)
            with patch.object(Policy, "load", side_effect=AssertionError("No policy may load")):
                with self.assertRaises(RuntimeError):
                    score_prm_file(source, output, root/"config.json", "cpu", reward_factory=factory)
                self.assertTrue(output.with_suffix(".partial.json").exists())
                score_prm_file(source, output, root/"config.json", "cpu", reward_factory=factory)
            result = json.loads(output.read_text())
            self.assertEqual(calls, ["0","1","1"])
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertTrue(result["metadata"]["prm_scored"])
            for old,new in zip(rows,result["rows"]):
                self.assertEqual(old["response"],new["response"])
                self.assertEqual(old["kl"],new["kl"])
                self.assertEqual(new["prm_reward"],0.8)
            self.assertFalse(output.with_suffix(".partial.json").exists())
            curve = frontier([output])
            self.assertEqual(curve["points"][0]["splits"]["gsm8k"]["correct"],1)
            different = copy.deepcopy(result)
            different["metadata"]["evaluation_input_hash"] = "another-panel"
            (root/"different.json").write_text(json.dumps(different))
            with self.assertRaises(ValueError):
                frontier([output, root/"different.json"])
            cfg.prm_revision = "different"
            (root/"config.json").write_text(json.dumps(cfg.to_dict()))
            with self.assertRaises(ValueError):
                score_prm_file(source, root/"other.json", root/"config.json", reward_factory=factory)

    def test_telemetry_storage_and_csv_flatten(self):
        from src.rewardmaxxing.telemetry import storage_sample, gpu_sample
        from scripts.export_metrics import flatten
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            root = Path(directory)
            adapter = root/"step-000001"/"adapter"
            adapter.mkdir(parents=True)
            (adapter/"adapter_model.safetensors").write_bytes(b"1234")
            (adapter.parent/"training_state.pt").write_bytes(b"12")
            cache = root/"cache.sqlite"
            cache.write_bytes(b"123")
            storage = storage_sample(root,cache)
            self.assertEqual(storage["adapter_bytes"]["step-000001"],4)
            self.assertEqual(storage["optimizer_bytes"]["step-000001"],2)
            self.assertEqual(storage["unexpected_full_model_files"],[])
            (adapter/"model.safetensors").write_bytes(b"x")
            self.assertEqual(len(storage_sample(root,cache)["unexpected_full_model_files"]),1)
        with patch("subprocess.check_output", return_value="0, 75, 40, 5000, 24000, 100\n"):
            self.assertEqual(gpu_sample()[0]["gpu_util_percent"],75)
        self.assertEqual(flatten({"step":2,"cache":{"hits":5}}), {"step":2,"cache/hits":5})

    def test_twin_subset_identity_and_fixed_instance_selection(self):
        from scripts.prepare_twins import select_instances
        from src.evaluate import load_twins
        rows = [{"original_id":1, "instance":i, "original_question":"Original",
                 "question":f"Variant {i}", "answer":"#### 3"} for i in (2,0,1)]
        selected = select_instances(rows, "main", 1)
        self.assertEqual(selected[0]["instance"],0)
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            path = Path(directory)/"twins.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in [selected[0],{**selected[0],"subset":"p1"}]))
            twins = load_twins(path,[{"id":"test:1","question":"Original"}])
            self.assertEqual(len({t["id"] for t in twins}),2)
            self.assertEqual({t["split"] for t in twins},{"gsm_symbolic/main","gsm_symbolic/p1"})


if __name__ == "__main__":
    began = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SmokeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"success": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "seconds": round(time.perf_counter()-began,3), "device":"cpu",
              "torch":torch.__version__, "transformers":transformers.__version__,
              "peft":peft.__version__, "pretrained_weights_downloaded":False,
              "external_api_requests":0}
    (ROOT/".tmp"/"smoke-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    faulthandler.cancel_dump_traceback_later()
    sys.exit(0 if result.wasSuccessful() else 1)
