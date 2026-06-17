// SignalGraph saved Neo4j Browser templates

// community_evidence: Community report inspection with member methods, papers, repos, and claims.
MATCH path = (member)-[:BELONGS_TO_COMMUNITY]->(community:Community)
WHERE $community_name = "" OR toLower(community.name) CONTAINS toLower($community_name)
RETURN path,
       community.name AS community,
       community.level AS level,
       labels(member) AS member_labels,
       coalesce(member.name, member.title, member.full_name, member.text) AS member
ORDER BY community, member
LIMIT $limit;

// conflicts: Contradictory claim pairs for review.
MATCH path = (left:Claim)-[r:CONTRADICTS]->(right:Claim)
RETURN path,
       left.text AS left_claim,
       right.text AS right_claim,
       r.confidence AS confidence,
       r.reason AS reason
ORDER BY confidence DESC
LIMIT $limit;

// counts: Typed label and relationship counts for local graph inspection.
CALL {
  MATCH (n)
  UNWIND labels(n) AS label
  RETURN "node" AS kind, label AS name, count(*) AS count
  UNION ALL
  MATCH ()-[r]->()
  RETURN "relationship" AS kind, type(r) AS name, count(*) AS count
}
RETURN kind, name, count
ORDER BY kind, name;

// duplicate_review: Low-confidence possible duplicate candidates for human review.
MATCH path = (left)-[r:POSSIBLE_DUPLICATE]->(right)
RETURN path,
       left.id AS left_id,
       right.id AS right_id,
       r.score AS score,
       r.signals AS signals,
       r.review_required AS review_required
ORDER BY score DESC
LIMIT $limit;

// evidence_paths: Paper/Repo -> Claim -> Source span typed evidence chains.
MATCH path = (source)-[:CLAIMS]->(claim:Claim)-[:SUPPORTED_BY]->(chunk:DocumentChunk)
WHERE source:Paper OR source:Repo
RETURN path,
       labels(source) AS source_labels,
       source.id AS source_id,
       claim.text AS claim,
       claim.confidence AS confidence,
       chunk.source_url AS source_url,
       chunk.source_span AS source_span
ORDER BY confidence DESC
LIMIT $limit;

// method_repo_evidence: Paper-method-repo paths with supporting claim evidence.
MATCH path = (paper:Paper)-[:INTRODUCES|USES_METHOD]->(method:Method)<-[:IMPLEMENTS|USES_METHOD]-(repo:Repo)
OPTIONAL MATCH (repo)-[:CLAIMS]->(claim:Claim)-[:SUPPORTED_BY]->(chunk:DocumentChunk)
WHERE toLower(method.name) CONTAINS toLower($method_name)
RETURN path,
       paper.title AS paper,
       method.name AS method,
       repo.full_name AS repo,
       claim.text AS claim,
       chunk.source_url AS source_url
LIMIT $limit;

// structured_repo_lookup: Structured lookup for repos implementing papers after a year with optional organization filter.
MATCH path = (repo:Repo)-[:IMPLEMENTS|USES_METHOD]->(target)
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
LIMIT $limit;
