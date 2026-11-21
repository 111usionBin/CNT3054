"""Read a single completed local run through MCP stdio."""
import json
import sys
from collections import Counter
from pathlib import Path

from mcp.server.fastmcp import FastMCP

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = data["records"]
mcp = FastMCP("CNT3054 Week 13")


@mcp.tool()
def lookup_utterance(utterance_id: str) -> dict:
    """발화 ID로 전사 원문, 시각, 각 모델의 라벨을 조회한다."""
    return {"mode": data["meta"]["mode"], "rows": [r for r in rows if r["utterance_id"] == utterance_id]}


@mcp.tool()
def find_disagreements(task: str = "stage") -> list[str]:
    """두 모델이 다르게 판정한 발화 ID를 찾는다. task는 speaker, emotion, stage 중 하나다."""
    if task not in ("speaker", "emotion", "stage"):
        raise ValueError("speaker, emotion, stage 중 하나를 입력하세요.")
    by_id = {}
    for row in rows:
        by_id.setdefault(row["utterance_id"], {})[row["model"]] = row[task]
    return [uid for uid, results in by_id.items() if len(set(results.values())) > 1]


@mcp.tool()
def speaker_emotions(model: str, speaker: str) -> dict:
    """선택한 모델이 그 화자에게 귀속한 발화의 감정 건수를 반환한다."""
    counts = Counter(r["emotion"] for r in rows if r["model"] == model and r["speaker"] == speaker)
    return {"model": model, "speaker": speaker, "n": sum(counts.values()), "counts": dict(counts)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
