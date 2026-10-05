"""CNT3054 Week 6: offline TF-IDF + logistic-regression sentiment lab.

The supplied reviews are synthetic teaching examples, not audience research.
Default: compare on validation only. Add --evaluate-test for a final evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import re
import unicodedata

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE / "data" / "synthetic_reviews.csv"
SEED = 42
LABELS = [0, 1]  # 0=negative, 1=positive; there is no neutral class.
CANDIDATES = {"unigram": (1, 1), "unigram_bigram": (1, 2)}


def normalize_text(text: str) -> str:
    """Unify Unicode and whitespace; keep negation, punctuation and original words."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_key(text: str) -> str:
    """Duplicates differing only in case/spacing/punctuation share one key."""
    return " ".join(re.findall(r"\w+", normalize_text(text).lower()))


def load_data(path: str | Path = DEFAULT_DATA) -> tuple[pd.DataFrame, dict]:
    """Validate human labels, remove exact normalized duplicates, keep groups.

    Required: text,label. Optional: review_id,group_id,domain,source_type.
    Never turn missing/neutral labels into a binary label automatically.
    """
    data = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = {"text", "label"} - set(data.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {sorted(missing)}")
    original_rows = len(data)
    data["text"] = data["text"].map(normalize_text)
    if data["text"].eq("").any():
        raise ValueError("빈 text가 있습니다. 원문을 확인하고 해당 행을 정리하세요.")
    data["label"] = data["label"].str.strip()
    if not data["label"].isin(["0", "1"]).all():
        raise ValueError("label은 0(부정) 또는 1(긍정)이어야 합니다. 중립/빈 라벨은 별도 검토하세요.")
    data["label"] = data["label"].astype(int)
    data["_text_key"] = data["text"].map(text_key)
    if data["_text_key"].eq("").any():
        raise ValueError("문자/숫자가 없는 text가 있습니다. 분석 가능 여부를 검토하세요.")
    if data.groupby("_text_key")["label"].nunique().gt(1).any():
        raise ValueError("같은 정규화 문장에 서로 다른 label이 있습니다. 라벨 기준을 먼저 점검하세요.")
    data = data.drop_duplicates("_text_key").copy().reset_index(drop=True)
    if "review_id" not in data:
        data["review_id"] = [f"ROW{i + 1:05}" for i in range(len(data))]
    if data["review_id"].eq("").any() or data["review_id"].duplicated().any():
        raise ValueError("review_id는 비어 있지 않고 각 행마다 달라야 합니다.")
    if "group_id" not in data:
        data["group_id"] = data["_text_key"]
    if data["group_id"].str.strip().eq("").any():
        raise ValueError("group_id가 비어 있습니다. 관련 문장의 그룹을 지정하세요.")
    data["group_id"] = data["group_id"].str.strip()
    if set(data["label"]) != {0, 1}:
        raise ValueError("학습에는 긍정과 부정 두 라벨이 모두 필요합니다.")
    group_counts = data.groupby("label")["group_id"].nunique()
    if group_counts.min() < 5:
        raise ValueError("각 라벨이 최소 5개의 서로 다른 그룹에 있어야 합니다. 더 많은 자료를 준비하세요.")
    metadata = {
        "input_rows": original_rows, "deduplicated_rows": len(data),
        "duplicates_removed": original_rows - len(data),
        "groups": int(data["group_id"].nunique()),
        "source_types": sorted(data["source_type"].unique().tolist())
        if "source_type" in data else ["user_provided_unverified"],
    }
    return data, metadata


def split_data(data: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """About 60/20/20; similar examples stay together and labels are stratified.

    One of five folds is test (20%). One of the remaining four is validation
    (25% of 80% = 20%). With uneven real-data groups, proportions are approximate.
    """
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    develop_idx, test_idx = next(outer.split(data["text"], data["label"], data["group_id"]))
    develop = data.iloc[develop_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(develop["text"], develop["label"], develop["group_id"]))
    splits = {
        "train": develop.iloc[train_idx].copy(),
        "validation": develop.iloc[val_idx].copy(),
        "test": data.iloc[test_idx].copy(),
    }
    for name, frame in splits.items():
        if set(frame["label"]) != {0, 1}:
            raise ValueError(f"{name}에 두 라벨이 모두 필요합니다. 그룹/라벨 분포와 자료 규모를 점검하세요.")
    return splits


def make_model(ngram_range: tuple[int, int] = (1, 1), seed: int = SEED) -> Pipeline:
    """Pipeline learns BOTH vocabulary/IDF and classifier from training only."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",  # Keep one-character Korean words: 안, 못.
            ngram_range=ngram_range, min_df=1, lowercase=True,
        )),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
    ])


def score_predictions(y_true, y_pred) -> dict:
    """Positive-class P/R/F1 and macro-F1 are explicitly distinguished."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_positive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
    }


def compare_models(train: pd.DataFrame, validation: pd.DataFrame, seed: int = SEED):
    """No test argument: model selection cannot read the held-out test set."""
    models, records = {}, []
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train[["text"]], train["label"])
    baseline_pred = baseline.predict(validation[["text"]])
    records.append({"model": "majority_baseline", **score_predictions(validation["label"], baseline_pred)})
    for name, ngrams in CANDIDATES.items():
        model = make_model(ngrams, seed)
        model.fit(train["text"], train["label"])
        models[name] = model
        records.append({"model": name, **score_predictions(validation["label"], model.predict(validation["text"]))})
    # Stable tie-break: first candidate (unigram) is simpler.
    selected = max(records[1:], key=lambda row: row["f1_macro"])["model"]
    return models, baseline, pd.DataFrame(records), selected


def prediction_table(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in ["review_id", "group_id", "domain", "text", "label"] if c in frame]
    result = frame[columns].copy()
    result["prediction"] = model.predict(frame["text"])
    positive_index = list(model.classes_).index(1)
    result["p_positive"] = model.predict_proba(frame["text"])[:, positive_index]
    result["correct"] = result["label"] == result["prediction"]
    result["error_type"] = "correct"
    result.loc[(result["label"] == 0) & (result["prediction"] == 1), "error_type"] = "FP"
    result.loc[(result["label"] == 1) & (result["prediction"] == 0), "error_type"] = "FN"
    # A useful warning for unseen domains, not an automatic neutral label.
    result["known_features"] = model.named_steps["tfidf"].transform(frame["text"]).getnnz(axis=1)
    return result.reset_index(drop=True)


def feature_weights(model: Pipeline) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if list(classifier.classes_) != [0, 1]:
        raise ValueError("계수 해석은 classes_ == [0, 1]일 때만 사용합니다.")
    return pd.DataFrame({
        "feature": model.named_steps["tfidf"].get_feature_names_out(),
        "weight_toward_positive": classifier.coef_[0],
    }).sort_values("weight_toward_positive").reset_index(drop=True)


def write_csv(frame: pd.DataFrame, path: Path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_plots(predictions: pd.DataFrame, weights: pd.DataFrame, output: Path, prefix: str):
    """No GUI needed; use Korean fonts if installed, otherwise numbered features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from sklearn.metrics import ConfusionMatrixDisplay

    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next((f for f in ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"] if f in installed), None)
    with plt.rc_context({"font.family": font or "DejaVu Sans", "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(5.4, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            predictions["label"], predictions["prediction"], labels=LABELS,
            display_labels=["negative (0)", "positive (1)"], cmap="Blues",
            colorbar=False, ax=ax,
        )
        ax.set_title(f"{prefix}: held-out predictions")
        fig.tight_layout()
        fig.savefig(output / f"{prefix}_confusion_matrix.png", dpi=160)
        plt.close(fig)
        if prefix != "validation":
            return
        top = pd.concat([weights.head(8), weights.tail(8)]).drop_duplicates("feature").copy()
        top["plot_label"] = top["feature"] if font else [f"feature {i+1:02}" for i in range(len(top))]
        write_csv(top, output / "top_features.csv")
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.barh(top["plot_label"], top["weight_toward_positive"], color=["#207b8d" if w < 0 else "#dd713f" for w in top["weight_toward_positive"]])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Logistic coefficient: negative < 0 < positive")
        ax.set_title("Training-only learned features (not causal effects)")
        fig.tight_layout()
        fig.savefig(output / "feature_weights.png", dpi=160)
        plt.close(fig)


def run_experiment(csv_path=DEFAULT_DATA, output_dir=None, evaluate_test=False, plots=True):
    """Validate, split, train, compare, then optionally evaluate the fixed winner."""
    data, data_meta = load_data(csv_path)
    splits = split_data(data)
    models, baseline, scores, selected = compare_models(splits["train"], splits["validation"])
    model = models[selected]  # No refit on validation: the same model is tested.
    output = Path(output_dir) if output_dir else BASE / "outputs" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"출력 폴더가 비어 있지 않습니다. 새 경로를 지정하세요: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(scores, output / "validation_scores.csv")
    validation = prediction_table(model, splits["validation"])
    weights = feature_weights(model)
    write_csv(validation, output / "validation_predictions.csv")
    write_csv(validation.loc[~validation["correct"]], output / "validation_errors.csv")
    write_csv(weights, output / "feature_weights.csv")
    manifest = pd.concat([frame[["review_id", "group_id"]].assign(split=name) for name, frame in splits.items()])
    write_csv(manifest, output / "split_assignments.csv")
    metadata = {
        "data": data_meta, "data_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "seed": SEED, "split_method": "StratifiedGroupKFold: 5 folds then 4 folds",
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "split_label_counts": {name: {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()} for name, frame in splits.items()},
        "selection_metric": "validation f1_macro; ties prefer unigram",
        "selected_model": selected, "selected_ngram_range": list(CANDIDATES[selected]),
        "fit_scope": "train only (including vocabulary and IDF); no validation refit",
        "positive_label": 1, "decision_rule": "sklearn predict; probability ties use class 0",
        "test_evaluated": bool(evaluate_test),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "warning": "Synthetic educational scores do not estimate real audience or domain-transfer accuracy. Probabilities are not sentiment intensity or guaranteed correctness.",
    }
    if plots:
        save_plots(validation, weights, output, "validation")
    if evaluate_test:
        test = prediction_table(model, splits["test"])
        write_csv(test, output / "test_predictions.csv")
        write_csv(test.loc[~test["correct"]], output / "test_errors.csv")
        baseline_test = baseline.predict(splits["test"][["text"]])
        test_scores = pd.DataFrame([
            {"model": "majority_baseline", **score_predictions(test["label"], baseline_test)},
            {"model": selected, **score_predictions(test["label"], test["prediction"])},
        ])
        write_csv(test_scores, output / "test_scores.csv")
        metadata["test_scores"] = test_scores.to_dict(orient="records")
        metadata["test_confusion_matrix_labels_0_1"] = confusion_matrix(test["label"], test["prediction"], labels=LABELS).tolist()
        report = classification_report(test["label"], test["prediction"], labels=LABELS, target_names=["negative (0)", "positive (1)"], zero_division=0, digits=4)
        (output / "test_classification_report.txt").write_text(report, encoding="utf-8")
        if plots:
            save_plots(test, weights, output, "test")
    (output / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, scores, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Week 6: TF-IDF + 로지스틱 회귀 (기본: 검증 세트만 평가)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA, help="text,label 열이 있는 UTF-8 CSV")
    parser.add_argument("--output-dir", type=Path, help="비어 있는 출력 폴더; 생략하면 새 시간표시 폴더")
    parser.add_argument("--evaluate-test", action="store_true", help="모델 선택 종료 후 고정 모델의 테스트 평가")
    parser.add_argument("--no-plots", action="store_true", help="PNG 없이 CSV와 JSON만 저장")
    args = parser.parse_args(argv)
    try:
        output, scores, meta = run_experiment(args.csv, args.output_dir, args.evaluate_test, not args.no_plots)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(1, f"입력/설정 확인: {exc}\n")
    print("[DATA]", ", ".join(meta["data"]["source_types"]))
    print("[SPLIT]", meta["split_sizes"])
    print(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("[SELECTED]", meta["selected_model"], "(validation macro-F1)")
    if args.evaluate_test:
        print("[FINAL TEST]", json.dumps(meta["test_scores"], ensure_ascii=False))
        print("[CAUTION] 테스트 결과를 보고 다시 모델을 고르면 새 독립 평가 자료가 필요합니다.")
    else:
        print("[TEST] 미평가. 비교/수정이 끝난 뒤에만 --evaluate-test를 사용하세요.")
    print("[CAUTION] 합성 예제의 점수는 실제 관객 반응 예측 성능이 아닙니다.")
    print("[SAVED]", output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
