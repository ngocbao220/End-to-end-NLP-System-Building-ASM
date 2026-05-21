from __future__ import annotations

import argparse
import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path


def normalize_answer(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, reference: str) -> tuple[float, float]:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall), recall


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def best_metric(prediction: str, references: list[str]) -> tuple[float, float, float]:
    scores = [(exact_match(prediction, ref), *f1_score(prediction, ref)) for ref in references]
    return max(scores, key=lambda item: (item[0], item[1], item[2]))


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate system outputs with SQuAD-style EM, F1 and recall.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    predictions = read_lines(args.predictions)
    references = [line.split(";") for line in read_lines(args.references)]
    if len(predictions) != len(references):
        raise ValueError(f"Prediction/reference length mismatch: {len(predictions)} vs {len(references)}")

    rows = []
    total_em = total_f1 = total_recall = 0.0
    questions = read_lines(args.questions) if args.questions else [""] * len(predictions)
    for idx, (question, prediction, refs) in enumerate(zip(questions, predictions, references), start=1):
        em, f1, recall = best_metric(prediction, refs)
        total_em += em
        total_f1 += f1
        total_recall += recall
        rows.append(
            {
                "index": idx,
                "question": question,
                "prediction": prediction,
                "references": refs,
                "exact_match": em,
                "f1": f1,
                "answer_recall": recall,
            }
        )

    n = len(predictions)
    result = {
        "count": n,
        "exact_match": total_em / n if n else 0.0,
        "f1": total_f1 / n if n else 0.0,
        "answer_recall": total_recall / n if n else 0.0,
        "examples": rows,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
