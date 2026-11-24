"""Run the week 12 → week 13 team workflow in the active Python environment."""
import asyncio
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from contextlib import ExitStack
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from dotenv import load_dotenv

import lab


def newest_transcript():
    outputs = lab.BASE.parent / "week12" / "outputs"
    candidates = sorted(outputs.glob("*/utterances.json"))
    if not candidates:
        raise ValueError(f"{outputs} 아래에 실행시각/utterances.json이 없습니다. --input으로 전사 파일을 지정하세요.")
    return candidates[-1]


def prepare_labels(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "speakers.json"
    speakers = dict.fromkeys((r["speaker_id"] for r in rows), "")
    if path.exists():
        saved = lab.read_json(path)
        if saved.keys() != speakers.keys():
            raise ValueError(f"{path}의 화자 목록이 전사와 다릅니다. --labels 경로를 확인하세요.")
        speakers.update(saved)
    for speaker, role in speakers.items():
        if isinstance(role, str) and 1 <= len(role.strip()) <= 120:
            continue
        while True:
            try:
                role = input(f"{speaker} 역할 소개 (1~120자): ").strip()
            except EOFError:
                raise ValueError("터미널에서 역할 소개를 입력하거나 --labels로 작성한 폴더를 지정하세요.") from None
            if 1 <= len(role) <= 120:
                break
        speakers[speaker] = role
        lab.write_json(path, speakers)
    gold = folder / "gold.csv"
    if not gold.exists():
        lab.save_csv(gold, ["utterance_id", *lab.TASKS], ({"utterance_id": r["utterance_id"]} for r in rows))
    return speakers


def local_bert(stack):
    url = urlsplit(os.getenv("BERT_API_URL", "http://127.0.0.1:8013/predict"))
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"} or url.path != "/predict":
        return
    def ready():
        try:
            with urlopen(f"http://{url.netloc}/openapi.json", timeout=1) as response:
                return "/predict" in json.load(response).get("paths", {})
        except (URLError, OSError, ValueError):
            return False
    if ready():
        return
    print("한국어 BERT를 시작합니다. 첫 실행에는 모델 다운로드 시간이 걸립니다.", flush=True)
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "bert_api:app", "--host",
                                url.hostname, "--port", str(url.port or 80)], cwd=lab.BASE)
    def stop():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    stack.callback(stop)
    deadline = time.monotonic() + 900
    while not ready():
        if process.poll() is not None or time.monotonic() > deadline:
            raise ValueError("BERT 시작에 실패했습니다. 위 서버 메시지와 requirements-bert.txt 설치 상태를 확인하세요.")
        time.sleep(1)


def execute(args):
    from mcp_demo import query
    load_dotenv(lab.BASE / ".env")
    source = (args.input or newest_transcript()).resolve()
    rows = lab.load_input(source)
    if args.limit < 1:
        raise ValueError("--limit은 1 이상이어야 합니다.")
    print(f"선택한 전사: {source}", flush=True)
    outputs = lab.BASE / "outputs"
    folder = outputs / "teams" / lab.digest(rows)[:16]
    legacy = outputs / "team01"
    checkpoint = legacy / "checkpoint.json"
    old_signature = lab.read_json(checkpoint)["signature"] if checkpoint.exists() else None
    rubric = lab.read_json(lab.BASE / "codebook.json")

    def signature(profiles):
        return lab.run_identity(False, rows, profiles, rubric, list(lab.MODELS), args.limit, {})[1]

    labels = args.labels
    if labels is None:
        labels = folder / "labels"
        prior = lab.BASE / "team_labels"
        if old_signature and (prior / "speakers.json").is_file():
            if signature(lab.read_json(prior / "speakers.json")) == old_signature:
                labels = prior
    print(f"팀 평가 폴더: {labels.resolve()}", flush=True)
    with lab.single_writer(labels):
        profiles = prepare_labels(labels, rows)
    current = signature(profiles)
    out = legacy if current == old_signature else folder / current[:16]
    run_args = Namespace(cmd="run", input=source, profiles=labels / "speakers.json",
                         gold=labels / "gold.csv", out=out, backend="both", limit=args.limit)
    with lab.single_writer(out), ExitStack() as stack:
        started = False
        def before_infer(model):
            nonlocal started
            if model == "bert" and not started:
                local_bert(stack)
                started = True
        lab.run(run_args, before_infer)
        ids = [r["utterance_id"] for r in rows[:args.limit]]
        asyncio.run(query(out / "results.json", "U0006" if "U0006" in ids else ids[0],
                          "SPEAKER_00" if "SPEAKER_00" in profiles else next(iter(profiles))))
