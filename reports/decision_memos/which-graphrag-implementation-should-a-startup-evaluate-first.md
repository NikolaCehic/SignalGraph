# Decision Memo: Which GraphRAG implementation should a startup evaluate first?

## Recommendation

Evaluate the linked implementation first, but require a small reproduction pass and repo-health review before adoption.

## Answer

The decision-grade answer combines community report breadth with local evidence chains: GraphRAG and agent memory systems can connect papers, repositories, datasets, and evaluation benchmarks for production assistants.; GraphRAG uses entity graphs, community detection, and community reports to improve answers for global questions over a corpus. The method is useful when evidence requires multi-ho...; Agent memory systems manage short-term and long-term context for language agents. The approach highlights production risks around memory freshness, retrieval policy, and evaluatio...

## Why

SignalGraph routed the query to drift retrieval and ranked evidence by semantic/lexical relevance, graph path quality, source quality, freshness, confidence, and evidence strength.

## Evidence

- `chunk:297194fc05a0`: GraphRAG and agent memory systems can connect papers, repositories, datasets, and evaluation benchmarks for production assistants. (https://www.semanticscholar.org/paper/S2-GRAPHRAG-MEMORY)
- `chunk:fc6e5b8d6191`: GraphRAG uses entity graphs, community detection, and community reports to improve answers for global questions over a corpus. The method is useful when evidence requires multi-hop relationships across papers, claims, and source chunks. (https://arxiv.org/abs/2404.16130)
- `repo:example/signalgraph-agent-memory`: Agent memory systems manage short-term and long-term context for language agents. The approach highlights production risks around memory freshness, retrieval policy, and evaluation of persisted context. Agent Memory Persistent or retrieved memory patterns for... (https://github.com/example/signalgraph-agent-memory)
- `repo:microsoft/graphrag`: A data pipeline and transformation suite for GraphRAG with community detection, reports, indexing, and retrieval workflows. A data pipeline and transformation suite for GraphRAG with community detection, reports, indexing, and retrieval workflows. A data pipe... (https://github.com/microsoft/graphrag)
- `community:agent-architecture`: A data pipeline and transformation suite for GraphRAG with community detection, reports, indexing, and retrieval workflows. A data pipeline and transformation suite for GraphRAG with community detection, reports, indexing, and retrieval workflows. graphrag mi... (local://signalgraph/community)
- `claim:5b33c60ba8dd`: GraphRAG uses entity graphs, community detection, and community reports to improve answers for global questions over a corpus. (https://arxiv.org/abs/2404.16130)

## Evidence Chains

- claim:b31740484e82 -> BELONGS_TO_COMMUNITY -> community:agent-architecture -> <-BELONGS_TO_COMMUNITY -> chunk:297194fc05a0
- paper:sample:graphrag -> CLAIMS -> claim:5b33c60ba8dd -> SUPPORTED_BY -> chunk:fc6e5b8d6191
- chunk:9300a3545ee0 -> MENTIONS -> method:agent-memory -> <-USES_METHOD -> repo:example/signalgraph-agent-memory
- claim:b31740484e82 -> SUPPORTED_BY -> chunk:83ff5028526f -> <-HAS_CHUNK -> repo:microsoft/graphrag
- claim:b31740484e82 -> <-CLAIMS -> repo:microsoft/graphrag -> IMPLEMENTS -> paper:sample:rag-eval -> BELONGS_TO_COMMUNITY -> community:agent-architecture
- paper:sample:memgpt -> CLAIMS -> claim:d12728f4e06b -> <-CONTRADICTS -> claim:5b33c60ba8dd
- claim:b31740484e82 -> BELONGS_TO_COMMUNITY -> community:agent-architecture -> <-BELONGS_TO_COMMUNITY -> chunk:297194fc05a0
- paper:sample:graphrag -> CLAIMS -> claim:5b33c60ba8dd -> SUPPORTED_BY -> chunk:fc6e5b8d6191
- community:agent-architecture -> <-BELONGS_TO_COMMUNITY -> chunk:251a3df8a74c -> <-SUPPORTED_BY -> claim:dec599dc88b9
- chunk:9300a3545ee0 -> MENTIONS -> method:agent-memory -> <-USES_METHOD -> repo:example/signalgraph-agent-memory

## Risks And Missing Evidence

- Retrieved evidence includes cautionary or limitation language; treat recommendation as conditional.
- No explicit benchmark or dataset node was retrieved for the top evidence chain.

## Vector-Only vs GraphRAG

- Vector-only faithfulness estimate: 0.657
- GraphRAG faithfulness estimate: 1.0
- Vector-only evidence-chain completeness: 0.143
- GraphRAG evidence-chain completeness: 1.0

## Next Checks

- Open the saved Cypher evidence-path query in Neo4j Browser.
- Inspect raw source records for the cited spans.
- Add benchmark or dataset extraction for the target method.
