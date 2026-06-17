from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CypherTemplate:
    name: str
    description: str
    cypher: str
    parameters: dict[str, Any]


TEMPLATES: dict[str, CypherTemplate] = {
    "evidence_paths": CypherTemplate(
        name="evidence_paths",
        description="Paper/Repo -> Claim -> Source span typed evidence chains.",
        parameters={"limit": 25},
        cypher="""MATCH path = (source)-[:CLAIMS]->(claim:Claim)-[:SUPPORTED_BY]->(chunk:DocumentChunk)
WHERE source:Paper OR source:Repo
RETURN path,
       labels(source) AS source_labels,
       source.id AS source_id,
       claim.text AS claim,
       claim.confidence AS confidence,
       chunk.source_url AS source_url,
       chunk.source_span AS source_span
ORDER BY confidence DESC
LIMIT $limit""",
    ),
    "method_repo_evidence": CypherTemplate(
        name="method_repo_evidence",
        description="Paper-method-repo paths with supporting claim evidence.",
        parameters={"method_name": "GraphRAG", "limit": 25},
        cypher="""MATCH path = (paper:Paper)-[:INTRODUCES|USES_METHOD]->(method:Method)<-[:IMPLEMENTS|USES_METHOD]-(repo:Repo)
OPTIONAL MATCH (repo)-[:CLAIMS]->(claim:Claim)-[:SUPPORTED_BY]->(chunk:DocumentChunk)
WHERE toLower(method.name) CONTAINS toLower($method_name)
RETURN path,
       paper.title AS paper,
       method.name AS method,
       repo.full_name AS repo,
       claim.text AS claim,
       chunk.source_url AS source_url
LIMIT $limit""",
    ),
    "structured_repo_lookup": CypherTemplate(
        name="structured_repo_lookup",
        description="Structured lookup for repos implementing papers after a year with optional organization filter.",
        parameters={"year": 2024, "organization": "", "limit": 50},
        cypher="""MATCH path = (repo:Repo)-[:IMPLEMENTS|USES_METHOD]->(target)
WHERE (target:Paper OR target:Method)
OPTIONAL MATCH (target:Paper)-[:AUTHORED_BY]->(author:Author)-[:AFFILIATED_WITH]->(org:Organization)
WHERE ($organization = "" OR toLower(org.name) CONTAINS toLower($organization))
  AND (target:Method OR coalesce(target.published_at, "") >= toString($year))
RETURN path,
       repo.full_name AS repo,
       labels(target) AS target_labels,
       coalesce(target.title, target.name) AS target,
       collect(DISTINCT org.name) AS organizations
ORDER BY repo
LIMIT $limit""",
    ),
    "counts": CypherTemplate(
        name="counts",
        description="Typed label and relationship counts for local graph inspection.",
        parameters={},
        cypher="""CALL {
  MATCH (n)
  UNWIND labels(n) AS label
  RETURN "node" AS kind, label AS name, count(*) AS count
  UNION ALL
  MATCH ()-[r]->()
  RETURN "relationship" AS kind, type(r) AS name, count(*) AS count
}
RETURN kind, name, count
ORDER BY kind, name""",
    ),
    "community_evidence": CypherTemplate(
        name="community_evidence",
        description="Community report inspection with member methods, papers, repos, and claims.",
        parameters={"community_name": "", "limit": 25},
        cypher="""MATCH path = (member)-[:BELONGS_TO_COMMUNITY]->(community:Community)
WHERE $community_name = "" OR toLower(community.name) CONTAINS toLower($community_name)
RETURN path,
       community.name AS community,
       community.level AS level,
       labels(member) AS member_labels,
       coalesce(member.name, member.title, member.full_name, member.text) AS member
ORDER BY community, member
LIMIT $limit""",
    ),
    "conflicts": CypherTemplate(
        name="conflicts",
        description="Contradictory claim pairs for review.",
        parameters={"limit": 25},
        cypher="""MATCH path = (left:Claim)-[r:CONTRADICTS]->(right:Claim)
RETURN path,
       left.text AS left_claim,
       right.text AS right_claim,
       r.confidence AS confidence,
       r.reason AS reason
ORDER BY confidence DESC
LIMIT $limit""",
    ),
    "duplicate_review": CypherTemplate(
        name="duplicate_review",
        description="Low-confidence possible duplicate candidates for human review.",
        parameters={"limit": 50},
        cypher="""MATCH path = (left)-[r:POSSIBLE_DUPLICATE]->(right)
RETURN path,
       left.id AS left_id,
       right.id AS right_id,
       r.score AS score,
       r.signals AS signals,
       r.review_required AS review_required
ORDER BY score DESC
LIMIT $limit""",
    ),
}


def list_templates() -> list[CypherTemplate]:
    return [TEMPLATES[name] for name in sorted(TEMPLATES)]


def get_template(name: str) -> CypherTemplate:
    try:
        return TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(TEMPLATES))
        raise ValueError(f"unknown Cypher template {name!r}; available: {available}") from exc


def inspect_template(name: str) -> dict[str, Any]:
    template = get_template(name)
    return {
        "name": template.name,
        "description": template.description,
        "parameters": template.parameters,
        "cypher": template.cypher,
    }


def saved_evidence_queries() -> str:
    sections = ["// SignalGraph saved Neo4j Browser templates"]
    for template in list_templates():
        sections.append("")
        sections.append(f"// {template.name}: {template.description}")
        sections.append(template.cypher + ";")
    return "\n".join(sections) + "\n"
