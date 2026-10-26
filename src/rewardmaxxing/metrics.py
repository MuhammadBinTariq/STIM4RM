from collections import defaultdict
import math
import random
import re
import statistics


def clean_solution(text, question=""):
    text = text.replace(question, "") if question else text
    text = re.sub(r"<<.*?>>", "", text)
    return text.replace("####", "")


def metric_tokens(text):
    return re.findall(r"\w+|[+\-*/=]", text.lower())


def lcs_length(tokens, reference_masks):
    """Exact token LCS using bit-parallel dynamic programming."""
    state = 0
    for token in tokens:
        x = reference_masks.get(token, 0) | state
        state = x & ~(x - ((state << 1) | 1))
    return state.bit_count()


class TrainingOverlap:
    def __init__(self, references):
        self.references = []
        self.ngrams = {4: set(), 8: set()}
        for text in references:
            tokens = metric_tokens(clean_solution(text))
            masks = defaultdict(int)
            for i, token in enumerate(tokens):
                masks[token] |= 1 << i
            self.references.append((len(tokens), masks))
            for n in self.ngrams:
                self.ngrams[n].update(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def score(self, text, question=""):
        tokens = metric_tokens(clean_solution(text, question))
        rouge = max((2*lcs_length(tokens, masks)/(len(tokens)+length)
                     for length, masks in self.references if len(tokens)+length), default=0.0)
        result = {"max_train_rouge_l_f1": rouge}
        for n, reference in self.ngrams.items():
            grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            result[f"train_{n}gram_coverage"] = sum(g in reference for g in grams)/len(grams) if grams else None
        return result


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def correlation(x, y):
    if len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    xx, yy = [v-mx for v in x], [v-my for v in y]
    den = math.sqrt(sum(v*v for v in xx)*sum(v*v for v in yy))
    return sum(a*b for a,b in zip(xx,yy))/den if den else None


def summarize(rows):
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    results = {}
    original = {r["original_id"]: r for r in rows if r["split"] == "gsm8k"}
    for split, subset in by_split.items():
        metrics = {"count": len(subset)}
        for key in ("correct", "format_valid", "kl", "length", "truncated", "prm_reward",
                    "max_train_rouge_l_f1", "train_4gram_coverage", "train_8gram_coverage",
                    "stim_rho", "stim_positive", "stim_valid_fraction"):
            metrics[key] = mean([r.get(key) for r in subset])
        scored = [r for r in subset if r.get("prm_reward") is not None]
        metrics["prm_accuracy_correlation"] = correlation([r["prm_reward"] for r in scored],
                                                         [r["correct"] for r in scored])
        metrics["high_prm_wrong_rate"] = mean([float(r["prm_reward"] >= 0.9 and not r["correct"]) for r in scored])
        if split != "gsm8k":
            paired = [r for r in subset if r["original_id"] in original]
            metrics["paired_count"] = len(paired)
            metrics["paired_accuracy_drop"] = mean([original[r["original_id"]]["correct"]-r["correct"] for r in paired])
            correct_original = [r for r in paired if original[r["original_id"]]["correct"]]
            metrics["original_correct_twin_wrong"] = mean([1-r["correct"] for r in correct_original])
        results[split] = metrics
    return results


def paired_comparison(baseline_runs, treatment_runs, bootstrap=2000, seed=1729):
    if len(baseline_runs) != len(treatment_runs) or not baseline_runs:
        raise ValueError("Supply equally many paired seeds")
    if bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    seeds = [run["metadata"]["seed"] for run in baseline_runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each paired run must use a distinct seed")
    paired = []
    for baseline, treatment in zip(baseline_runs, treatment_runs):
        for key in ("seed", "model_id", "model_revision", "dataset_revision", "eval_max_new_tokens",
                    "training_manifest_hash", "evaluation_input_hash", "reward", "stim_audit",
                    "split", "decoding", "metric_tokenizer", "prm_scored", "prm_quantized",
                    "prm_id", "prm_revision", "prm_base_revision", "training_protocol", "checkpoint_name"):
            if baseline["metadata"][key] != treatment["metadata"][key]:
                raise ValueError(f"Mismatched paired evaluation metadata: {key}")
        if baseline["metadata"]["lambda"] != 0 or not (treatment["metadata"]["lambda"] or 0) > 0:
            raise ValueError("Comparison requires a trained lambda=0 baseline and lambda>0 treatment")
        a, b = ({r["id"]:r for r in run["rows"]} for run in (baseline,treatment))
        if len(a) != len(baseline["rows"]) or len(b) != len(treatment["rows"]):
            raise ValueError("Duplicate evaluation row IDs")
        if a.keys() != b.keys():
            raise ValueError("Paired evaluation sets differ")
        for key in a:
            if any(a[key][field] != b[key][field] for field in ("question", "reference", "split", "original_id")):
                raise ValueError("Question/reference mismatch")
        paired.append([(a[k],b[k]) for k in sorted(a)])
    splits = sorted({a["split"] for pair in paired for a,b in pair})
    rng = random.Random(seed)
    output = {}
    for split in splits:
        for metric in ("correct", "stim_rho", "max_train_rouge_l_f1", "kl"):
            seed_clusters = []
            for pair in paired:
                clusters = defaultdict(list)
                for a,b in pair:
                    if a["split"] == split and a.get(metric) is not None and b.get(metric) is not None:
                        clusters[a["original_id"]].append(b[metric]-a[metric])
                seed_clusters.append(dict(clusters))
            if any(not clusters for clusters in seed_clusters):
                output[f"{split}/{metric}"] = {"delta":None, "reason":"missing measurements in one or more seeds"}
                continue
            estimates = [mean([v for values in clusters.values() for v in values]) for clusters in seed_clusters]
            draws = []
            for _ in range(bootstrap):
                estimates_draw = []
                for clusters in rng.choices(seed_clusters, k=len(seed_clusters)):
                    sampled = rng.choices(list(clusters), k=len(clusters))
                    estimates_draw.append(mean([v for k in sampled for v in clusters[k]]))
                draws.append(mean(estimates_draw))
            draws.sort()
            output[f"{split}/{metric}"] = {"delta":mean(estimates), "seed_deltas":estimates,
                "seed_std":statistics.stdev(estimates) if len(estimates)>1 else None,
                "ci95":[draws[int(bootstrap*.025)],draws[min(bootstrap-1,int(bootstrap*.975))]],
                "paired_observations_per_seed":[sum(map(len,c.values())) for c in seed_clusters]}
    return output
