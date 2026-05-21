from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Python 3.11.0rc1 misses this API, but recent PyTorch imports expect it.
if not hasattr(sys, "get_int_max_str_digits"):
    def _get_int_max_str_digits() -> int:
        return 0

    sys.get_int_max_str_digits = _get_int_max_str_digits  # type: ignore[attr-defined]
if not hasattr(sys, "set_int_max_str_digits"):
    def _set_int_max_str_digits(maxdigits: int) -> None:
        return None

    sys.set_int_max_str_digits = _set_int_max_str_digits  # type: ignore[attr-defined]

import torch


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "did",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "whom",
    "with",
    "duoc",
    "la",
    "nam",
    "vao",
    "ve",
    "cua",
    "co",
    "bao",
    "nhieu",
    "truong",
    "dai",
    "hoc",
}


@dataclass(frozen=True)
class Fact:
    id: str
    split: str
    category: str
    source: str
    question: str
    answer: str
    context: str

    @property
    def retrieval_text(self) -> str:
        return f"{self.question} {self.answer} {self.context} {self.category}"


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    normalized = strip_accents(text).lower()
    tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", normalized)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def load_facts(path: Path, split: str | None = None) -> list[Fact]:
    facts: list[Fact] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            fact = Fact(
                id=raw["id"],
                split=raw.get("split", ""),
                category=raw.get("category", ""),
                source=raw.get("source", ""),
                question=raw["question"],
                answer=raw["answer"],
                context=raw.get("context", ""),
            )
            if split is None or fact.split == split:
                facts.append(fact)
    return facts


class SentenceTransformerRetriever:
    def __init__(
        self,
        facts: Iterable[Fact],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        lexical_rerank_weight: float = 0.18,
        local_files_only: bool = False,
    ) -> None:
        self.facts = list(facts)
        if not self.facts:
            raise ValueError("The knowledge base is empty.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.lexical_rerank_weight = lexical_rerank_weight
        self.fact_question_tokens = [set(tokenize(fact.question)) for fact in self.facts]
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        texts = [fact.retrieval_text for fact in self.facts]
        self.doc_embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[Fact, float]]:
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = self.doc_embeddings @ query_embedding
        query_tokens = set(tokenize(query))
        if query_tokens and self.lexical_rerank_weight:
            lexical_scores = [
                len(query_tokens & fact_tokens) / len(query_tokens)
                for fact_tokens in self.fact_question_tokens
            ]
            scores = scores + torch.tensor(lexical_scores, device=scores.device) * self.lexical_rerank_weight
        limit = min(top_k, len(self.facts))
        values, indices = torch.topk(scores, k=limit)
        return [(self.facts[index.item()], float(score.item())) for score, index in zip(values, indices)]


class RAGSystem:
    def __init__(
        self,
        facts: Iterable[Fact],
        min_score: float = 0.25,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        lexical_rerank_weight: float = 0.18,
        local_files_only: bool = False,
    ) -> None:
        self.retriever = SentenceTransformerRetriever(
            facts,
            model_name=model_name,
            device=device,
            lexical_rerank_weight=lexical_rerank_weight,
            local_files_only=local_files_only,
        )
        self.min_score = min_score

    def answer(self, question: str) -> str:
        hits = self.retriever.search(question, top_k=3)
        if not hits:
            return "unknown"
        best, score = hits[0]
        if score < self.min_score:
            return "unknown"
        return best.answer

    def answer_with_evidence(self, question: str) -> dict[str, object]:
        hits = self.retriever.search(question, top_k=3)
        answer = "unknown"
        if hits and hits[0][1] >= self.min_score:
            answer = hits[0][0].answer
        return {
            "question": question,
            "answer": answer,
            "evidence": [
                {
                    "id": fact.id,
                    "score": round(score, 4),
                    "source": fact.source,
                    "context": fact.context,
                }
                for fact, score in hits
            ],
        }


def read_questions(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sentence-transformers dense-retrieval RAG QA system.")
    parser.add_argument("--facts", type=Path, default=Path("data/raw/facts.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/test/questions.txt"))
    parser.add_argument("--output", type=Path, default=Path("system_outputs/system_output_1.txt"))
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--lexical-rerank-weight", type=float, default=0.18)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    facts = load_facts(args.facts, split=split)
    try:
        rag = RAGSystem(
            facts,
            min_score=args.min_score,
            model_name=args.model,
            device=args.device,
            lexical_rerank_weight=args.lexical_rerank_weight,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        print(
            "Failed to initialize sentence-transformers RAG. "
            "Install dependencies with `python3 -m pip install -r requirements.txt` "
            "and ensure the embedding model can be downloaded or is cached locally.",
            file=sys.stderr,
        )
        raise exc
    questions = read_questions(args.questions)
    answers = [rag.answer(question) for question in questions]
    write_lines(args.output, answers)

    if args.evidence_output:
        args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
        with args.evidence_output.open("w", encoding="utf-8") as f:
            for question in questions:
                f.write(json.dumps(rag.answer_with_evidence(question), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
