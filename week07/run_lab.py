"""Week 7 entry point for an existing uv course environment."""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import warnings
from datetime import datetime, timezone

import pandas as pd
import lab


def client_from_terminal():
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        # Never silently fall back to echoed input (e.g. the Output panel).
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                key = getpass.getpass("NVIDIA API key (hidden): ")
            except getpass.GetPassWarning:
                raise RuntimeError("숨김 입력이 가능한 VS Code 통합 터미널에서 실행하세요.") from None
    return lab.make_client(key)


def announce_results(folder):
    print(f"\n결과 폴더: {folder}")
    print("제출물: 실제 결과를 해석한 A4 2쪽 내외 보고서 PDF 1개")
    return Path(folder)


def analyze(use_nvidia=False, use_bert=False, model=lab.MODEL_ID):
    train, probe = lab.load_data()
    print(f"학습: {len(train)}, 진단: {len(probe)}")
    result, _ = lab.run_baselines(train, probe)
    revision = None
    # BERT runs first so a failed model download does not waste API calls.
    if use_bert:
        print("BERT 다운로드/CPU 추론 중...")
        predictions, revision = lab.run_bert(probe)
        result = pd.concat([result.reset_index(drop=True), predictions], axis=1)
    if use_nvidia:
        print("NVIDIA 진단 문장 16개를 전송합니다.")
        with client_from_terminal() as client:
            result = result.merge(lab.run_nvidia(client, probe, model),
                                  on="review_id", validate="one_to_one")
    scores = lab.compare_metrics(result)
    print("\n공통 긍정/부정 12문장 지표 (실패/보류 포함)")
    print(scores.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    folder = lab.save_results(result, scores, model=model, extra={
        "entrypoint": "run_lab.py", "nvidia_executed": use_nvidia,
        "bert_executed": use_bert, "bert_revision": revision,
        "methods_executed": scores.method.tolist(),
    })
    announce_results(folder)
    return folder


def embeddings():
    train, _ = lab.load_data()
    w2v, neighbors = lab.word2vec_demo(train)
    texts = ["이 영화는 재미있다.", "이 작품은 정말 즐겁다.",
             "이 영화는 재미있지 않다.", "공연은 일곱 시에 시작한다."]
    vectors, similarities = lab.sentence_similarity(texts)
    ids = [f"S{i+1}" for i in range(len(texts))]
    matrix = pd.DataFrame(similarities, index=ids, columns=ids)
    print(f"Word2Vec '영화는': {w2v.wv['영화는'].shape}")
    print(neighbors.to_string(index=False))
    print(f"\n문장 임베딩: {vectors.shape}")
    print(matrix.to_string(float_format=lambda x: f"{x:.3f}"))
    folder = lab.ROOT / "outputs" / datetime.now(timezone.utc).strftime("embeddings_%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True)
    neighbors.to_csv(folder / "word2vec_neighbors.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(folder / "sentence_similarity.csv", encoding="utf-8-sig")
    (folder / "sentences.json").write_text(json.dumps({
        "model": lab.EMBED_ID, "texts": dict(zip(ids, texts)),
        "sentence_shape": list(vectors.shape), "word2vec_shape": [32],
        "note": "Synthetic examples, similarity is not sentiment probability",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    announce_results(folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CNT3054 Week 7 · VS Code 터미널 실습")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("baseline", help="사전 + TF-IDF 실행 및 결과 CSV 저장 (API 없음)")
    sub.add_parser("prompt", help="zero-shot 지시문 확인 (API 없음)")
    smoke = sub.add_parser("smoke", help="숨김 키 입력 후 NVIDIA 1문장 연결 점검")
    smoke.add_argument("--model", default=lab.MODEL_ID)
    full = sub.add_parser("analyze", help="기준선과 선택한 모델을 같은 16문장으로 비교")
    full.add_argument("--nvidia", action="store_true", help="NVIDIA API 호출")
    full.add_argument("--bert", action="store_true", help="로컬 CPU BERT 추가")
    full.add_argument("--model", default=lab.MODEL_ID)
    sub.add_parser("embeddings", help="Word2Vec와 문장 임베딩 실행 (확장 패키지 필요)")
    args = parser.parse_args(argv)
    try:
        if args.command == "prompt":
            print(lab.PROMPT)
        elif args.command == "smoke":
            with client_from_terminal() as client:
                answer = lab.classify_one(client, "이 영화는 재미있지 않다.", model=args.model)
            print(json.dumps(answer, ensure_ascii=False, indent=2))
            return 0 if answer["status"] == "ok" else 1
        elif args.command == "embeddings":
            embeddings()
        elif args.command == "baseline":
            analyze()
        else:
            if not (args.nvidia or args.bert):
                parser.error("analyze에는 --nvidia 또는 --bert를 지정하세요. 기준선만은 baseline입니다.")
            analyze(args.nvidia, args.bert, args.model)
    except (RuntimeError, ValueError, ModuleNotFoundError, OSError) as exc:
        print(f"실행 중단: {exc}", file=sys.stderr)
        print("README의 문제 해결을 확인하세요. API 키는 코드에 쓰지 마세요.", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\n입력을 취소했습니다.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
