from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .embeddings import DEFAULT_EMBEDDING_DIMENSIONS
from .models import GraphArtifact, GraphEdge, GraphNode


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")),
            password=os.environ.get("NEO4J_PASSWORD", ""),
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        )

    def safe_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["password"] = "<set>" if self.password else "<unset>"
        return payload


@dataclass(frozen=True)
class Neo4jBatch:
    kind: str
    name: str
    count: int
    cypher: str
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Neo4jLoadPlan:
    dry_run: bool
    config: dict[str, str]
    constraints: list[str]
    fulltext_indexes: list[str]
    vector_indexes: list[str]
    batches: list[Neo4jBatch]

    def to_dict(self, include_rows: bool = False) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "config": self.config,
            "constraints": self.constraints,
            "fulltext_indexes": self.fulltext_indexes,
            "vector_indexes": self.vector_indexes,
            "batch_counts": {batch.name: batch.count for batch in self.batches},
            "batches": [
                {
                    "kind": batch.kind,
                    "name": batch.name,
                    "count": batch.count,
                    "cypher": batch.cypher,
                    **({"rows": batch.rows} if include_rows else {}),
                }
                for batch in self.batches
            ],
        }


def constraint_statements() -> list[str]:
    return [
        "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (n:Paper) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (n:Author) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT organization_id IF NOT EXISTS FOR (n:Organization) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT repo_id IF NOT EXISTS FOR (n:Repo) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT repo_full_name IF NOT EXISTS FOR (n:Repo) REQUIRE n.full_name IS UNIQUE",
        "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (n:Method) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT benchmark_id IF NOT EXISTS FOR (n:Benchmark) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (n:Dataset) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT model_id IF NOT EXISTS FOR (n:Model) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:DocumentChunk) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_hash IF NOT EXISTS FOR (n:DocumentChunk) REQUIRE n.hash IS UNIQUE",
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (n:Community) REQUIRE n.id IS UNIQUE",
    ]


def fulltext_index_statements() -> list[str]:
    return [
        "CREATE FULLTEXT INDEX signalgraph_paper_text IF NOT EXISTS FOR (n:Paper) ON EACH [n.title, n.abstract]",
        "CREATE FULLTEXT INDEX signalgraph_method_text IF NOT EXISTS FOR (n:Method) ON EACH [n.name, n.description, n.category]",
        "CREATE FULLTEXT INDEX signalgraph_repo_text IF NOT EXISTS FOR (n:Repo) ON EACH [n.full_name, n.name, n.description]",
        "CREATE FULLTEXT INDEX signalgraph_benchmark_text IF NOT EXISTS FOR (n:Benchmark) ON EACH [n.name, n.task, n.metric]",
        "CREATE FULLTEXT INDEX signalgraph_claim_chunk_text IF NOT EXISTS FOR (n:Claim|DocumentChunk) ON EACH [n.text, n.source_span, n.section]",
        "CREATE FULLTEXT INDEX signalgraph_community_text IF NOT EXISTS FOR (n:Community) ON EACH [n.name, n.summary, n.report]",
    ]


def vector_index_statements(dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> list[str]:
    config = "{`vector.dimensions`: %d, `vector.similarity_function`: 'cosine'}" % dimensions
    return [
        f"CREATE VECTOR INDEX signalgraph_chunk_embedding IF NOT EXISTS FOR (n:DocumentChunk) ON (n.embedding) OPTIONS {{indexConfig: {config}}}",
        f"CREATE VECTOR INDEX signalgraph_claim_embedding IF NOT EXISTS FOR (n:Claim) ON (n.embedding) OPTIONS {{indexConfig: {config}}}",
        f"CREATE VECTOR INDEX signalgraph_method_embedding IF NOT EXISTS FOR (n:Method) ON (n.embedding) OPTIONS {{indexConfig: {config}}}",
        f"CREATE VECTOR INDEX signalgraph_repo_readme_embedding IF NOT EXISTS FOR (n:Repo) ON (n.readme_embedding) OPTIONS {{indexConfig: {config}}}",
        f"CREATE VECTOR INDEX signalgraph_community_embedding IF NOT EXISTS FOR (n:Community) ON (n.embedding) OPTIONS {{indexConfig: {config}}}",
    ]


class Neo4jLoader:
    def __init__(self, config: Neo4jConfig | None = None, batch_size: int = 500):
        self.config = config or Neo4jConfig.from_env()
        self.batch_size = batch_size

    def plan(self, artifact: GraphArtifact, dry_run: bool = True) -> Neo4jLoadPlan:
        dimensions = _embedding_dimensions(artifact)
        return Neo4jLoadPlan(
            dry_run=dry_run,
            config=self.config.safe_dict(),
            constraints=constraint_statements(),
            fulltext_indexes=fulltext_index_statements(),
            vector_indexes=vector_index_statements(dimensions),
            batches=_batches(artifact),
        )

    def load(self, artifact: GraphArtifact, dry_run: bool = True) -> Neo4jLoadPlan:
        plan = self.plan(artifact, dry_run=dry_run)
        if dry_run:
            return plan
        if not self.config.password:
            raise ValueError("NEO4J_PASSWORD is required for live Neo4j loading; use dry_run=True for local checks")
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(self.config.uri, auth=(self.config.username, self.config.password))
        try:
            with driver.session(database=self.config.database) as session:
                for statement in [*plan.constraints, *plan.fulltext_indexes, *plan.vector_indexes]:
                    session.run(statement)
                for batch in plan.batches:
                    for rows in _chunk(batch.rows, self.batch_size):
                        session.run(batch.cypher, rows=rows)
        finally:
            driver.close()
        return plan


def _batches(artifact: GraphArtifact) -> list[Neo4jBatch]:
    batches: list[Neo4jBatch] = []
    nodes_by_label: dict[str, list[GraphNode]] = {}
    for node in artifact.nodes:
        label = _primary_label(node)
        nodes_by_label.setdefault(label, []).append(node)
    for label, nodes in sorted(nodes_by_label.items()):
        rows = [{"id": node.id, "properties": node.properties} for node in nodes]
        batches.append(
            Neo4jBatch(
                kind="node",
                name=f"nodes:{label}",
                count=len(rows),
                cypher=f"UNWIND $rows AS row MERGE (n:{_cypher_name(label)} {{id: row.id}}) SET n += row.properties",
                rows=rows,
            )
        )
    edges_by_type: dict[str, list[GraphEdge]] = {}
    for edge in artifact.edges:
        edges_by_type.setdefault(edge.type, []).append(edge)
    for rel_type, edges in sorted(edges_by_type.items()):
        rows = [
            {
                "id": edge.id,
                "start_id": edge.start_id,
                "end_id": edge.end_id,
                "properties": edge.properties,
            }
            for edge in edges
        ]
        batches.append(
            Neo4jBatch(
                kind="relationship",
                name=f"relationships:{rel_type}",
                count=len(rows),
                cypher=(
                    "UNWIND $rows AS row "
                    "MATCH (a {id: row.start_id}) "
                    "MATCH (b {id: row.end_id}) "
                    f"MERGE (a)-[r:{_cypher_name(rel_type)} {{id: row.id}}]->(b) "
                    "SET r += row.properties"
                ),
                rows=rows,
            )
        )
    return batches


def _embedding_dimensions(artifact: GraphArtifact) -> int:
    for node in artifact.nodes:
        value = node.properties.get("embedding_dimensions")
        if value:
            return int(value)
    return DEFAULT_EMBEDDING_DIMENSIONS


def _primary_label(node: GraphNode) -> str:
    for label in node.labels:
        if label != "Entity":
            return label
    return node.labels[0]


def _cypher_name(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch == "_")


def _chunk(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]
