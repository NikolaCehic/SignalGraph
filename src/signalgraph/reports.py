from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .answering import AnswerSynthesizer, compare
from .config import ProjectPaths
from .evaluation import EvalRunner
from .retrieval import Retriever
from .utils import slugify, write_json


class ReportWriter:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.paths.ensure()

    def decision_memo(self, query: str) -> dict[str, Path]:
        retriever = Retriever(self.paths)
        result = retriever.graph_aware(query)
        answer = AnswerSynthesizer().synthesize(result)
        comparison = compare(self.paths, query)
        slug = slugify(query, "decision-memo")
        md_path = self.paths.decision_memos_dir / f"{slug}.md"
        json_path = self.paths.artifacts_dir / f"{slug}-decision-memo.json"
        csv_path = self.paths.artifacts_dir / f"{slug}-citations.csv"
        md_path.write_text(_decision_memo_markdown(query, answer.to_dict(), comparison), encoding="utf-8")
        write_json(json_path, {"answer": answer.to_dict(), "comparison": comparison})
        _write_citations_csv(csv_path, answer.to_dict()["citations"])
        return {"markdown": md_path, "json": json_path, "csv": csv_path}

    def eval_summary(self) -> dict[str, Path]:
        EvalRunner(self.paths).run()
        return {
            "markdown": self.paths.reports_dir / "eval_summary.md",
            "json": self.paths.eval_results_path,
            "csv": self.paths.retrieval_comparison_path,
            "retrieval_markdown": self.paths.reports_dir / "retrieval_quality.md",
            "retrieval_csv": self.paths.reports_dir / "retrieval_quality.csv",
            "retrieval_json": self.paths.artifacts_dir / "retrieval_quality.json",
            "generation_markdown": self.paths.reports_dir / "generation_quality.md",
            "generation_csv": self.paths.reports_dir / "generation_quality.csv",
            "generation_json": self.paths.artifacts_dir / "generation_quality.json",
            "system_markdown": self.paths.reports_dir / "system_health.md",
            "system_json": self.paths.artifacts_dir / "system_health.json",
            "failure_markdown": self.paths.reports_dir / "failure_cases.md",
            "failure_csv": self.paths.reports_dir / "failure_cases.csv",
            "failure_json": self.paths.artifacts_dir / "failure_cases.json",
            "trace_jsonl": self.paths.traces_dir / "eval_query_traces.jsonl",
        }


def _decision_memo_markdown(query: str, answer: dict[str, Any], comparison: dict[str, Any]) -> str:
    lines = [
        f"# Decision Memo: {query}",
        "",
        "## Recommendation",
        "",
        answer["production_recommendation"],
        "",
        "## Answer",
        "",
        answer["answer"],
        "",
        "## Why",
        "",
        answer["reasoning"],
        "",
        "## Evidence",
        "",
    ]
    for citation in answer["citations"]:
        lines.append(f"- `{citation['node_id']}`: {citation['source_span']} ({citation['source_url']})")
    lines.extend(["", "## Evidence Chains", ""])
    for chain in answer["evidence_chain"]:
        lines.append(f"- {' -> '.join(chain)}")
    lines.extend(["", "## Risks And Missing Evidence", ""])
    for note in answer["conflicts_or_missing_evidence"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Vector-Only vs GraphRAG", ""])
    lines.append(f"- Vector-only faithfulness estimate: {comparison['faithfulness_estimate']['vector_only']}")
    lines.append(f"- GraphRAG faithfulness estimate: {comparison['faithfulness_estimate']['graph_rag']}")
    lines.append(f"- Vector-only evidence-chain completeness: {comparison['evidence_chain_completeness']['vector_only']}")
    lines.append(f"- GraphRAG evidence-chain completeness: {comparison['evidence_chain_completeness']['graph_rag']}")
    lines.extend(["", "## Next Checks", ""])
    for check in answer["next_checks"]:
        lines.append(f"- {check}")
    return "\n".join(lines) + "\n"


def _write_citations_csv(path: Path, citations: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["node_id", "labels", "source_url", "source_span", "score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for citation in citations:
            row = dict(citation)
            row["labels"] = ",".join(row.get("labels", []))
            writer.writerow(row)
