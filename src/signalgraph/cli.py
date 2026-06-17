from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .answering import AnswerSynthesizer, compare
from .community import write_community_report_artifacts
from .config import ProjectPaths
from .cypher_templates import inspect_template, list_templates
from .evaluation import BUILTIN_EVAL_QUESTIONS, EvalRunner
from .graph import GraphBuilder, load_graph
from .ingest import CorpusSizeControls, Ingestor
from .neo4j_loader import Neo4jLoader
from .reports import ReportWriter
from .retrieval import Retriever
from .sample_data import ensure_sample_corpus


SOURCE_CHOICES = ["arxiv", "openalex", "semantic_scholar", "github", "huggingface"]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = ProjectPaths(Path(args.root).expanduser().resolve()) if getattr(args, "root", None) else ProjectPaths.default()
    try:
        if not hasattr(args, "handler"):
            parser.print_help()
            return
        args.handler(args, paths)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"signalgraph: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signalgraph",
        description="SignalGraph engine CLI for ingestion, graph building, retrieval, evals, and reports.",
    )
    parser.add_argument("--root", help="Project root. Defaults to current working directory or SIGNALGRAPH_HOME.")
    subcommands = parser.add_subparsers(dest="command")

    sources = subcommands.add_parser("sources", help="Search public source APIs.")
    sources_sub = sources.add_subparsers(dest="sources_command")
    sources_search = sources_sub.add_parser("search", help="Search arXiv, OpenAlex, Semantic Scholar, GitHub, and Hugging Face.")
    sources_search.add_argument("--topic", required=True)
    sources_search.add_argument("--limit", type=int, default=25)
    sources_search.add_argument("--per-source-limit", type=int, help="Bound records requested from each selected source before total limit trimming.")
    sources_search.add_argument("--source", action="append", choices=SOURCE_CHOICES, help="Limit to one or more sources.")
    sources_search.set_defaults(handler=handle_sources_search)

    ingest = subcommands.add_parser("ingest", help="Run ingestion pipelines.")
    ingest_sub = ingest.add_subparsers(dest="ingest_command")
    ingest_run = ingest_sub.add_parser("run", help="Fetch, archive, normalize, and save source records.")
    ingest_run.add_argument("--topic", required=True)
    ingest_run.add_argument("--limit", type=int, default=25)
    ingest_run.add_argument("--per-source-limit", type=int, help="Bound records requested from each selected source before total limit trimming.")
    ingest_run.add_argument("--source", action="append", choices=SOURCE_CHOICES, help="Limit to one or more sources.")
    ingest_run.add_argument("--starter-targets", action="store_true", help="Apply roadmap starter upper bounds: 250 papers, 100 repos, 40 methods, 50 datasets/benchmarks/models, and 800 claims.")
    ingest_run.add_argument("--max-source-records", type=int, help="Maximum source records to normalize from this run.")
    ingest_run.add_argument("--max-papers", type=int, help="Maximum paper records to keep from this run.")
    ingest_run.add_argument("--max-authors", type=int, help="Maximum author records to keep from this run.")
    ingest_run.add_argument("--max-organizations", type=int, help="Maximum organization records to keep from this run.")
    ingest_run.add_argument("--max-repos", type=int, help="Maximum repo records to keep from this run.")
    ingest_run.add_argument("--max-repo-documents", type=int, help="Maximum README/docs/changelog records to keep from this run.")
    ingest_run.add_argument("--max-repo-releases", type=int, help="Maximum GitHub release records to keep from this run.")
    ingest_run.add_argument("--max-repo-issues", type=int, help="Maximum GitHub issue records to keep from this run.")
    ingest_run.add_argument("--max-benchmarks", type=int, help="Maximum benchmark records to keep from this run.")
    ingest_run.add_argument("--max-datasets", type=int, help="Maximum dataset records to keep from this run.")
    ingest_run.add_argument("--max-models", type=int, help="Maximum model records to keep from this run.")
    ingest_run.add_argument("--max-methods", type=int, help="Maximum method records to keep from this run.")
    ingest_run.add_argument("--max-chunks", type=int, help="Maximum document chunks to keep from this run.")
    ingest_run.add_argument("--max-claims", type=int, help="Maximum claim records to keep from this run.")
    ingest_run.add_argument("--sample", action="store_true", help="Use bundled deterministic sample corpus instead of live APIs.")
    ingest_run.set_defaults(handler=handle_ingest_run)

    graph = subcommands.add_parser("graph", help="Build and inspect graph artifacts.")
    graph_sub = graph.add_subparsers(dest="graph_command")
    graph_build = graph_sub.add_parser("build", help="Build typed graph JSON and Cypher artifacts.")
    graph_build.add_argument("--no-sample", action="store_true", help="Do not seed bundled sample data when no normalized corpus exists.")
    graph_build.set_defaults(handler=handle_graph_build)
    graph_stats = graph_sub.add_parser("stats", help="Print graph node and relationship counts.")
    graph_stats.set_defaults(handler=handle_graph_stats)
    graph_communities = graph_sub.add_parser("communities", help="Generate and inspect deterministic community reports.")
    graph_communities.add_argument("--limit", type=int, default=20)
    graph_communities.set_defaults(handler=handle_graph_communities)
    graph_lookup = graph_sub.add_parser("structured-lookup", help="Run a Cypher-template structured graph lookup locally.")
    graph_lookup.add_argument("--query", required=True)
    graph_lookup.add_argument("--limit", type=int, default=8)
    graph_lookup.set_defaults(handler=handle_graph_structured_lookup)
    graph_load = graph_sub.add_parser("load-neo4j", help="Load the graph into Neo4j or print a dry-run load plan.")
    graph_load.add_argument("--dry-run", action="store_true", default=True, help="Print constraints, indexes, and batch counts without connecting to Neo4j.")
    graph_load.add_argument("--execute", action="store_true", help="Execute against Neo4j using NEO4J_* environment variables.")
    graph_load.add_argument("--batch-size", type=int, default=500)
    graph_load.set_defaults(handler=handle_graph_load_neo4j)
    graph_templates = graph_sub.add_parser("cypher-template", help="Inspect Neo4j Cypher templates for evidence paths, lookups, and counts.")
    graph_templates.add_argument("--name", help="Template name to print. Omit to list templates.")
    graph_templates.set_defaults(handler=handle_graph_cypher_template)

    ask_parser = subcommands.add_parser("ask", help="Ask a graph-aware research-to-production question.")
    ask_parser.add_argument("--mode", choices=["auto", "vector", "local", "global", "drift", "hybrid", "structured_lookup"], default="auto")
    ask_parser.add_argument("query")
    ask_parser.set_defaults(handler=handle_ask)

    compare_parser = subcommands.add_parser("compare", help="Compare vector-only RAG against SignalGraph GraphRAG.")
    compare_parser.add_argument("query")
    compare_parser.set_defaults(handler=handle_compare)

    eval_parser = subcommands.add_parser("eval", help="Run evaluation suites.")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")
    eval_run = eval_sub.add_parser("run", help="Run the 70-question corpus across vector-only, hybrid, local, global, DRIFT-style, and best-route ablations.")
    eval_run.set_defaults(handler=handle_eval_run)
    eval_corpus = eval_sub.add_parser("corpus", help="Inspect the built-in 70-question evaluation corpus.")
    eval_corpus.add_argument("--list", action="store_true", help="List question IDs, categories, and queries.")
    eval_corpus.add_argument("--json", action="store_true", help="Print the corpus as JSON.")
    eval_corpus.set_defaults(handler=handle_eval_corpus)

    report_parser = subcommands.add_parser("report", help="Generate Markdown, JSON, and CSV reports.")
    report_sub = report_parser.add_subparsers(dest="report_command")
    decision = report_sub.add_parser("decision-memo", help="Generate a decision memo for a query.")
    decision.add_argument("--query", required=True)
    decision.set_defaults(handler=handle_report_decision_memo)
    eval_summary = report_sub.add_parser("eval-summary", help="Generate eval summary, retrieval quality, generation quality, system health, failure case, and trace artifacts.")
    eval_summary.set_defaults(handler=handle_report_eval_summary)

    return parser


def handle_sources_search(args: argparse.Namespace, paths: ProjectPaths) -> None:
    documents = Ingestor(paths).search(args.topic, limit=args.limit, source_names=args.source, per_source_limit=args.per_source_limit)
    print(f"Found {len(documents)} source records for topic: {args.topic}")
    for document in documents:
        title = (
            document.raw_payload.get("title")
            or document.raw_payload.get("display_name")
            or document.raw_payload.get("full_name")
            or document.raw_payload.get("modelId")
            or document.raw_payload.get("id")
            or document.raw_payload.get("name")
            or document.source_id
        )
        print(f"- [{document.source_name}] {title} :: {document.source_url}")


def handle_ingest_run(args: argparse.Namespace, paths: ProjectPaths) -> None:
    size_controls = _size_controls_from_args(args)
    if args.sample:
        ensure_sample_corpus(paths, force=True)
        if size_controls:
            from .storage import NormalizedStore

            store = NormalizedStore(paths)
            store.save(size_controls.apply(store.load()))
        stats = _corpus_stats(paths)
    else:
        stats = Ingestor(paths).run(
            args.topic,
            limit=args.limit,
            source_names=args.source,
            per_source_limit=args.per_source_limit,
            size_controls=size_controls,
        )
    print("Ingestion complete")
    _print_kv(stats)
    print(f"normalized_corpus: {paths.normalized_corpus_path}")


def handle_graph_build(args: argparse.Namespace, paths: ProjectPaths) -> None:
    artifact = GraphBuilder(paths).build(use_sample_if_empty=not args.no_sample)
    print("Graph artifact built")
    _print_kv(artifact.stats())
    print(f"graph_json: {paths.graph_artifact_path}")
    print(f"cypher_export: {paths.cypher_export_path}")
    print(f"evidence_queries: {paths.evidence_query_path}")


def handle_graph_stats(args: argparse.Namespace, paths: ProjectPaths) -> None:
    graph = load_graph(paths, build_if_missing=True)
    _print_kv(graph.stats())


def handle_graph_communities(args: argparse.Namespace, paths: ProjectPaths) -> None:
    artifact = GraphBuilder(paths).build(use_sample_if_empty=True)
    reports = write_community_report_artifacts(artifact, paths.graph_dir / "community_reports.json")
    print("Community reports generated")
    print(f"community_reports_json: {paths.graph_dir / 'community_reports.json'}")
    for report in reports[: args.limit]:
        print(f"- {report['id']} :: {report['name']} :: size={report['size']} :: confidence={report['confidence']}")
        print(f"  members: {', '.join(report['member_ids'][:6])}")


def handle_graph_structured_lookup(args: argparse.Namespace, paths: ProjectPaths) -> None:
    result = Retriever(paths).structured_lookup(args.query, limit=args.limit)
    print(f"query: {result.query}")
    print(f"mode: {result.mode}")
    print(f"cypher_template: {result.trace.get('cypher_template', '')}")
    print("parameters:")
    print(json.dumps(result.trace.get("parameters", {}), indent=2, sort_keys=True))
    print("results:")
    for candidate in result.candidates[: args.limit]:
        print(f"- {candidate.node_id} :: score={candidate.score} :: labels={','.join(candidate.labels)}")
        print(f"  path: {' -> '.join(candidate.path)}")


def handle_graph_load_neo4j(args: argparse.Namespace, paths: ProjectPaths) -> None:
    graph = load_graph(paths, build_if_missing=True)
    dry_run = not args.execute
    plan = Neo4jLoader(batch_size=args.batch_size).load(graph, dry_run=dry_run)
    payload = plan.to_dict(include_rows=False)
    print("Neo4j load dry-run" if dry_run else "Neo4j load complete")
    print("connection:")
    _print_kv(payload["config"], prefix="  ")
    print("constraints:")
    for statement in payload["constraints"]:
        print(f"- {statement}")
    print("fulltext_indexes:")
    for statement in payload["fulltext_indexes"]:
        print(f"- {statement}")
    print("vector_indexes:")
    for statement in payload["vector_indexes"]:
        print(f"- {statement}")
    print("batch_counts:")
    _print_kv(payload["batch_counts"], prefix="  ")


def handle_graph_cypher_template(args: argparse.Namespace, paths: ProjectPaths) -> None:
    if not args.name:
        print("Available Cypher templates:")
        for template in list_templates():
            print(f"- {template.name}: {template.description}")
        return
    payload = inspect_template(args.name)
    print(f"name: {payload['name']}")
    print(f"description: {payload['description']}")
    print("parameters:")
    print(json.dumps(payload["parameters"], indent=2, sort_keys=True))
    print("cypher:")
    print(payload["cypher"])


def handle_ask(args: argparse.Namespace, paths: ProjectPaths) -> None:
    result = Retriever(paths).retrieve(args.query, mode=args.mode)
    answer = AnswerSynthesizer().synthesize(result)
    print(f"route: {answer.route}")
    print(f"mode: {result.mode}")
    print(f"confidence: {answer.confidence}")
    print()
    print(answer.answer)
    print()
    print("citations:")
    for citation in answer.citations[:5]:
        print(f"- {citation['node_id']} :: {citation['source_url']} :: {citation['source_span']}")
    print()
    print("evidence_paths:")
    for chain in answer.evidence_chain[:5]:
        print(f"- {' -> '.join(chain)}")
    if answer.conflicts_or_missing_evidence:
        print()
        print("conflicts_or_missing_evidence:")
        for note in answer.conflicts_or_missing_evidence:
            print(f"- {note}")
    print()
    print(f"production_recommendation: {answer.production_recommendation}")
    print("next_checks:")
    for check in answer.next_checks:
        print(f"- {check}")


def handle_compare(args: argparse.Namespace, paths: ProjectPaths) -> None:
    payload = compare(paths, args.query)
    print(f"query: {payload['query']}")
    print()
    print("vector_only:")
    print(f"- confidence: {payload['vector_only']['confidence']}")
    print(f"- answer: {payload['vector_only']['answer']}")
    print()
    print("graph_rag:")
    print(f"- confidence: {payload['graph_rag']['confidence']}")
    print(f"- answer: {payload['graph_rag']['answer']}")
    print()
    print("metrics:")
    print(json.dumps({
        "faithfulness_estimate": payload["faithfulness_estimate"],
        "evidence_chain_completeness": payload["evidence_chain_completeness"],
        "latency_ms": payload["latency_ms"],
        "cost_usd_estimate": payload["cost_usd_estimate"],
    }, indent=2, sort_keys=True))


def handle_eval_run(args: argparse.Namespace, paths: ProjectPaths) -> None:
    payload = EvalRunner(paths).run()
    print(f"questions: {payload['summary']['question_count']}")
    print(f"rows: {payload['summary']['row_count']}")
    print(f"ablations: {', '.join(payload['summary']['ablations'])}")
    print("categories:")
    _print_kv(payload["summary"]["category_counts"], prefix="  ")
    print("summary:")
    _print_kv(payload["summary"])
    print("graph_metrics:")
    _print_kv({key: payload["summary"]["graph_metrics"][key] for key in payload["graph_metrics"]}, prefix="  ")
    print(f"eval_json: {paths.eval_results_path}")
    print(f"retrieval_csv: {paths.retrieval_comparison_path}")
    print(f"eval_summary_md: {paths.reports_dir / 'eval_summary.md'}")
    print(f"trace_jsonl: {paths.traces_dir / 'eval_query_traces.jsonl'}")


def handle_eval_corpus(args: argparse.Namespace, paths: ProjectPaths) -> None:
    questions = BUILTIN_EVAL_QUESTIONS
    rows = [asdict(question) for question in questions]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    counts = Counter(question.category for question in questions)
    print(f"questions: {len(questions)}")
    print("categories:")
    _print_kv(dict(sorted(counts.items())), prefix="  ")
    print(f"corpus_path_when_run: {paths.eval_dir / 'signalgraph_eval_corpus.json'}")
    if args.list:
        print("items:")
        for question in questions:
            print(f"- {question.id} :: {question.category} :: {question.query}")


def handle_report_decision_memo(args: argparse.Namespace, paths: ProjectPaths) -> None:
    outputs = ReportWriter(paths).decision_memo(args.query)
    print("Decision memo generated")
    for key, path in outputs.items():
        print(f"{key}: {path}")


def handle_report_eval_summary(args: argparse.Namespace, paths: ProjectPaths) -> None:
    outputs = ReportWriter(paths).eval_summary()
    print("Eval summary generated")
    for key, path in outputs.items():
        print(f"{key}: {path}")


def _print_kv(value: Any, prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                print(f"{prefix}{key}:")
                _print_kv(item, prefix="  ")
            else:
                print(f"{prefix}{key}: {item}")
    else:
        print(value)


def _corpus_stats(paths: ProjectPaths) -> dict[str, int]:
    from .storage import NormalizedStore

    corpus = NormalizedStore(paths).load()
    return {
        "source_records": len(corpus.source_records),
        "papers": len(corpus.papers),
        "authors": len(corpus.authors),
        "organizations": len(corpus.organizations),
        "repos": len(corpus.repos),
        "repo_documents": len(corpus.repo_documents),
        "repo_releases": len(corpus.repo_releases),
        "repo_issues": len(corpus.repo_issues),
        "benchmarks": len(corpus.benchmarks),
        "datasets": len(corpus.datasets),
        "models": len(corpus.models),
        "methods": len(corpus.methods),
        "claims": len(corpus.claims),
        "chunks": len(corpus.chunks),
        "communities": len(corpus.communities),
        "extraction_quarantine": len(corpus.extraction_quarantine),
    }


def _size_controls_from_args(args: argparse.Namespace) -> CorpusSizeControls | None:
    controls = CorpusSizeControls.starter_targets() if args.starter_targets else CorpusSizeControls()
    for attr in [
        "max_source_records",
        "max_papers",
        "max_authors",
        "max_organizations",
        "max_repos",
        "max_repo_documents",
        "max_repo_releases",
        "max_repo_issues",
        "max_benchmarks",
        "max_datasets",
        "max_models",
        "max_methods",
        "max_chunks",
        "max_claims",
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            setattr(controls, attr, value)
    values = [getattr(controls, attr) for attr in controls.__dataclass_fields__]
    return controls if any(value is not None for value in values) else None


if __name__ == "__main__":
    main()
