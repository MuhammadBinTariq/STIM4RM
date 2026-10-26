import hashlib
import random


def question_hash(question):
    return hashlib.sha256(" ".join(question.split()).encode()).hexdigest()


def validate_splits(train, dev, test):
    sets = [{question_hash(x["question"]) for x in rows} for rows in (train, dev, test)]
    for i in range(3):
        if len(sets[i]) != len((train, dev, test)[i]):
            raise ValueError("Duplicate questions in a split")
        for j in range(i):
            if sets[i] & sets[j]:
                raise ValueError("Training/development/test question leakage")


def load_gsm8k(cfg):
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", revision=cfg.dataset_revision)
    train_pool = [{"id": f"train:{i}", **dict(row)} for i, row in enumerate(dataset["train"])]
    test = [{"id": f"test:{i}", **dict(row)} for i, row in enumerate(dataset["test"])]
    random.Random(cfg.split_seed).shuffle(train_pool)
    if cfg.train_size + cfg.dev_size > len(train_pool):
        raise ValueError("Requested train/dev sizes exceed official training split")
    dev = train_pool[:cfg.dev_size]
    train = train_pool[cfg.dev_size:cfg.dev_size + cfg.train_size]
    validate_splits(train, dev, test)
    manifest = {"source": "openai/gsm8k", "revision": cfg.dataset_revision,
                "train_ids": [x["id"] for x in train], "dev_ids": [x["id"] for x in dev],
                "test_ids": [x["id"] for x in test],
                "train_hashes": [question_hash(x["question"]) for x in train]}
    return train, dev, test, manifest


def batches(examples, cfg):
    result = []
    for epoch in range(cfg.epochs):
        order = list(examples)
        random.Random(cfg.seed + epoch).shuffle(order)
        result.extend(order[i:i+cfg.prompts_per_batch]
                      for i in range(0, len(order), cfg.prompts_per_batch))
    return result
