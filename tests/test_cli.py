from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_cli(*args, root=None):
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root / "src")
    command = [sys.executable, "-m", "signalgraph.cli"]
    if root is not None:
        command.extend(["--root", str(root)])
    command.extend(args)
    return subprocess.run(command, cwd=repo_root, env=env, text=True, capture_output=True, check=False)


def test_cli_help_lists_required_command_groups():
    result = run_cli("--help")
    assert result.returncode == 0
    for word in ["sources", "ingest", "graph", "ask", "compare", "eval", "report"]:
        assert word in result.stdout


def test_cli_nested_help_lists_required_subcommands():
    checks = [
        ("sources", "search", "--help"),
        ("ingest", "run", "--help"),
        ("graph", "build", "--help"),
        ("graph", "stats", "--help"),
        ("graph", "communities", "--help"),
        ("graph", "structured-lookup", "--help"),
        ("graph", "load-neo4j", "--help"),
        ("graph", "cypher-template", "--help"),
        ("eval", "run", "--help"),
        ("eval", "corpus", "--help"),
        ("report", "decision-memo", "--help"),
        ("report", "eval-summary", "--help"),
    ]
    for args in checks:
        result = run_cli(*args)
        assert result.returncode == 0, result.stderr


def test_cli_help_lists_expanded_source_choices_and_corpus_controls():
    sources = run_cli("sources", "search", "--help")
    assert sources.returncode == 0, sources.stderr
    assert "semantic_scholar" in sources.stdout
    assert "huggingface" in sources.stdout
    assert "--per-source-limit" in sources.stdout

    ingest = run_cli("ingest", "run", "--help")
    assert ingest.returncode == 0, ingest.stderr
    assert "semantic_scholar" in ingest.stdout
    assert "huggingface" in ingest.stdout
    for flag in ["--starter-targets", "--max-papers", "--max-repos", "--max-models", "--max-datasets", "--max-claims"]:
        assert flag in ingest.stdout


def test_cli_commands_run_without_network_credentials_or_neo4j(tmp_path):
    build = run_cli("graph", "build", root=tmp_path)
    assert build.returncode == 0, build.stderr
    assert "Graph artifact built" in build.stdout

    ask = run_cli("ask", "Which GraphRAG implementation should a startup evaluate first?", root=tmp_path)
    assert ask.returncode == 0, ask.stderr
    assert "route:" in ask.stdout
    assert "mode:" in ask.stdout
    assert "citations:" in ask.stdout
    assert "production_recommendation:" in ask.stdout

    communities = run_cli("graph", "communities", root=tmp_path)
    assert communities.returncode == 0, communities.stderr
    assert "Community reports generated" in communities.stdout

    hybrid = run_cli("ask", "--mode", "hybrid", "Compare GraphRAG and agent memory for production support automation.", root=tmp_path)
    assert hybrid.returncode == 0, hybrid.stderr
    assert "mode: hybrid" in hybrid.stdout

    structured = run_cli("ask", "--mode", "structured_lookup", "Which repos implement papers after 2024?", root=tmp_path)
    assert structured.returncode == 0, structured.stderr
    assert "mode: structured_lookup" in structured.stdout

    lookup = run_cli("graph", "structured-lookup", "--query", "Which repos implement papers after 2024?", root=tmp_path)
    assert lookup.returncode == 0, lookup.stderr
    assert "cypher_template: structured_repo_lookup" in lookup.stdout
    assert "results:" in lookup.stdout

    corpus = run_cli("eval", "corpus", root=tmp_path)
    assert corpus.returncode == 0, corpus.stderr
    assert "questions: 70" in corpus.stdout
    assert "adversarial/uncertainty" in corpus.stdout

    evaluation = run_cli("eval", "run", root=tmp_path)
    assert evaluation.returncode == 0, evaluation.stderr
    assert "graph_path_recall" in evaluation.stdout
