"""Run from repo root: python -m unittest discover -s week06/tests -v."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_sentiment as lab


class SentimentLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.info = lab.load_data()
        cls.splits = lab.split_data(cls.data)

    def write_fixture(self, frame, directory):
        p = Path(directory) / "fixture.csv"
        frame.to_csv(p, index=False, encoding="utf-8-sig")
        return p

    def test_synthetic_data_schema_and_balance(self):
        self.assertEqual(len(self.data), 120)
        self.assertEqual(self.data["label"].value_counts().to_dict(), {1: 60, 0: 60})
        self.assertEqual(self.data["group_id"].nunique(), 60)
        self.assertEqual(self.info["source_types"], ["synthetic_educational"])
        self.assertEqual(self.info["duplicates_removed"], 0)
        self.assertTrue(self.data.groupby("group_id")["label"].nunique().eq(2).all())

    def test_split_disjoint_groups_text_and_balanced_rows(self):
        self.assertEqual({n: len(f) for n, f in self.splits.items()}, {"train": 72, "validation": 24, "test": 24})
        frames = list(self.splits.values())
        for i, frame in enumerate(frames):
            self.assertEqual(frame["label"].value_counts().nunique(), 1)
            for other in frames[i + 1:]:
                for column in ["review_id", "group_id", "_text_key"]:
                    self.assertFalse(set(frame[column]) & set(other[column]))
        self.assertEqual(sum(map(len, frames)), len(self.data))

    def test_split_reproducible(self):
        again = lab.split_data(self.data)
        for name in self.splits:
            pd.testing.assert_frame_equal(self.splits[name], again[name])

    def test_vocabulary_and_idf_are_train_only(self):
        model = lab.make_model()
        model.fit(["좋다 전용학습단어", "싫다 부정학습단어"], [1, 0])
        vocabulary_before = dict(model.named_steps["tfidf"].vocabulary_)
        idf_before = model.named_steps["tfidf"].idf_.copy()
        model.predict(["검증전용유출단어"])
        self.assertNotIn("검증전용유출단어", vocabulary_before)
        self.assertEqual(vocabulary_before, model.named_steps["tfidf"].vocabulary_)
        self.assertTrue((idf_before == model.named_steps["tfidf"].idf_).all())

    def test_preserves_single_character_negation(self):
        analyzer = lab.make_model((1, 2)).named_steps["tfidf"].build_analyzer()
        self.assertIn("안", analyzer("안 좋다"))
        self.assertIn("안 좋다", analyzer("안 좋다"))

    def test_duplicate_normalization_and_conflict_check(self):
        with tempfile.TemporaryDirectory() as d:
            extra = self.data.iloc[[0]].drop(columns=["_text_key"]).copy()
            extra["review_id"] = "DUPLICATE"
            extra["text"] = "  " + extra["text"].str.replace(".", "!", regex=False) + "  "
            fixture = pd.concat([self.data.drop(columns=["_text_key"]), extra], ignore_index=True)
            clean, meta = lab.load_data(self.write_fixture(fixture, d))
            self.assertEqual(len(clean), 120)
            self.assertEqual(meta["duplicates_removed"], 1)
            fixture.loc[len(fixture) - 1, "label"] = 0
            with self.assertRaisesRegex(ValueError, "서로 다른 label"):
                lab.load_data(self.write_fixture(fixture, d))

    def test_invalid_input_is_rejected(self):
        cases = [
            ("text", "", "빈 text"),
            ("text", "!!!", "문자/숫자"),
            ("label", "neutral", "label은"),
            ("label", "", "label은"),
            ("group_id", "  ", "group_id"),
            ("review_id", "", "review_id"),
        ]
        for column, value, message in cases:
            with self.subTest(column=column, value=value), tempfile.TemporaryDirectory() as d:
                frame = self.data.drop(columns=["_text_key"]).astype(str).copy()
                frame.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    lab.load_data(self.write_fixture(frame, d))

    def test_one_class_or_too_few_groups_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for frame in [self.data[self.data.label == 1], self.data.head(8)]:
                with self.assertRaises(ValueError):
                    lab.load_data(self.write_fixture(frame, d))

    def test_plain_text_label_csv_is_supported(self):
        with tempfile.TemporaryDirectory() as d:
            clean, _ = lab.load_data(self.write_fixture(self.data[["text", "label"]], d))
            self.assertEqual(clean.review_id.nunique(), 120)
            self.assertEqual(clean.group_id.nunique(), 120)
            self.assertEqual(sum(len(f) for f in lab.split_data(clean).values()), 120)

    def test_metrics_match_hand_calculation(self):
        true = [0] * 12 + [1] * 12
        pred = [0] * 9 + [1] * 3 + [0] + [1] * 11
        scores = lab.score_predictions(true, pred)
        self.assertAlmostEqual(scores["accuracy"], 20 / 24)
        self.assertAlmostEqual(scores["precision_positive"], 11 / 14)
        self.assertAlmostEqual(scores["recall_positive"], 11 / 12)
        self.assertAlmostEqual(scores["f1_positive"], 22 / 26)
        self.assertEqual(confusion_matrix(true, pred, labels=[0, 1]).tolist(), [[9, 3], [1, 11]])

    def test_validation_only_never_predicts_test(self):
        seen = []
        original = lab.Pipeline.predict
        def audited_predict(model, text, **kwargs):
            seen.extend(list(text))
            return original(model, text, **kwargs)
        with tempfile.TemporaryDirectory() as d, patch.object(lab.Pipeline, "predict", audited_predict):
            output, scores, meta = lab.run_experiment(output_dir=Path(d) / "out", plots=False)
            self.assertFalse(meta["test_evaluated"])
            self.assertEqual(meta["selected_model"], "unigram")
            self.assertFalse(list(output.glob("test_*")))
            self.assertFalse(set(seen) & set(self.splits["test"]["text"]))
            self.assertEqual(len(scores), 3)

    def test_final_outputs_reproducible_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name in ["a", "b"]:
                out, _, meta = lab.run_experiment(output_dir=Path(d) / name, evaluate_test=True, plots=False)
                self.assertTrue(meta["test_evaluated"])
                self.assertEqual(meta["test_confusion_matrix_labels_0_1"], [[9, 3], [1, 11]])
                self.assertEqual(len(pd.read_csv(out / "test_predictions.csv")), 24)
                self.assertEqual(len(pd.read_csv(out / "test_errors.csv")), 4)
                self.assertTrue((out / "test_predictions.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
                paths.append(out)
            for filename in ["test_predictions.csv", "validation_scores.csv", "split_assignments.csv"]:
                self.assertEqual((paths[0] / filename).read_bytes(), (paths[1] / filename).read_bytes())
            with self.assertRaisesRegex(ValueError, "비어 있지"):
                lab.run_experiment(output_dir=paths[0], plots=False)

    def test_coefficients_and_unknown_text(self):
        model = lab.make_model().fit(self.splits["train"]["text"], self.splits["train"]["label"])
        weights = lab.feature_weights(model)
        self.assertEqual(len(weights), len(model.named_steps["tfidf"].vocabulary_))
        self.assertTrue(weights.weight_toward_positive.is_monotonic_increasing)
        unknown = pd.DataFrame({"review_id": ["NEW"], "text": ["검증전용미등록표현"], "label": [0]})
        prediction = lab.prediction_table(model, unknown)
        self.assertEqual(prediction.loc[0, "known_features"], 0)
        self.assertIn(prediction.loc[0, "prediction"], [0, 1])

    def test_plot_pngs_and_no_korean_font_fallback(self):
        from matplotlib import font_manager
        model = lab.make_model().fit(self.splits["train"]["text"], self.splits["train"]["label"])
        predictions = lab.prediction_table(model, self.splits["validation"])
        weights = lab.feature_weights(model)
        latin_fonts = [font for font in font_manager.fontManager.ttflist if font.name == "DejaVu Sans"]
        with tempfile.TemporaryDirectory() as d, patch.object(font_manager.fontManager, "ttflist", latin_fonts):
            output = Path(d)
            lab.save_plots(predictions, weights, output, "validation")
            for filename in ["validation_confusion_matrix.png", "feature_weights.png"]:
                self.assertTrue((output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            top = pd.read_csv(output / "top_features.csv")
            self.assertTrue(top["plot_label"].str.startswith("feature ").all())
            self.assertEqual(len(top), 16)

    def test_cli_invalid_file_returns_clear_error(self):
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as exit_ctx:
            lab.main(["--csv", "/this/file/does/not/exist.csv", "--no-plots"])
        self.assertEqual(exit_ctx.exception.code, 1)
        self.assertIn("입력/설정 확인", error.getvalue())


if __name__ == "__main__":
    unittest.main()
