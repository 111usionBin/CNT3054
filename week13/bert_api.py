"""Korean RoBERTa NLI over (dialogue window, label hypothesis) pairs."""
import os
from contextlib import asynccontextmanager
from threading import Lock

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.environ.get("BERT_MODEL_PATH", "Huffon/klue-roberta-base-nli")
REV = None if os.path.isdir(MODEL) else "3778d23ecb30a63babb17f5efb37b1493b08d975"
LOCK = Lock()


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    app.state.model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REV).eval()
    yield


app = FastAPI(lifespan=lifespan)


class Item(BaseModel):
    target: str
    previous: str
    following: str
    profiles: dict[str, str]
    rubric: dict


@app.post("/predict")
def predict(item: Item):
    rubric = item.rubric
    groups = {"speaker": {key: [sentence.strip() + "." for sentence in desc.split(".") if sentence.strip()]
                           for key, desc in item.profiles.items()}, **rubric["hypotheses"]}
    context = f"{item.previous} [{item.target}] {item.following}".strip()
    pairs = [(task, label, hypothesis) for task, labels in groups.items()
             for label, examples in labels.items() for hypothesis in examples]
    model = app.state.model
    ids = {k.upper(): v for k, v in model.config.label2id.items()}
    scores = {task: {label: 0.0 for label in labels} for task, labels in groups.items()}
    with LOCK, torch.inference_mode():
        for i in range(0, len(pairs), 8):
            batch = pairs[i:i + 8]
            hypotheses = [x[2] for x in batch]
            # The target is weighted more heavily than its neighbours.
            texts = [item.target] * len(batch) + [context] * len(batch)
            tokens = app.state.tokenizer(texts, hypotheses * 2, padding=True, truncation="only_first",
                                         max_length=480, return_token_type_ids=False, return_tensors="pt")
            logits = model(**tokens).logits
            # Relative support for entailment vs contradiction, as in zero-shot NLI.
            # Neutral evidence can make this uncertain; the top-two margin permits abstention.
            support = logits[:, [ids["CONTRADICTION"], ids["ENTAILMENT"]]].softmax(dim=-1)[:, 1]
            target_scores, context_scores = support.chunk(2)
            for (task, label, _), score in zip(batch, (0.8 * target_scores + 0.2 * context_scores).tolist()):
                scores[task][label] = max(scores[task][label], score)
    result = {}
    for task, by_label in scores.items():
        ranked = sorted(by_label, key=by_label.get, reverse=True)
        first = by_label[ranked[0]]
        second = by_label[ranked[1]] if len(ranked) > 1 else 0
        result[task] = ranked[0] if len(item.target.strip(" .,!?。")) > 2 and first >= .60 and first - second >= .08 else rubric["abstain"]
    return {**result, "scores": scores}
