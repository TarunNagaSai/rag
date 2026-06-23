# 07 · GraphRAG

> Code: `graph.py`

## The failure mode vectors can't fix

Vector search is **local**: it finds passages similar to your query. It struggles with
**global** and **multi-hop** questions that require connecting facts across many documents:

- "Which customers renewed **and** opened an SSO ticket?" (a join across two facts)
- "Who reported the bug that delayed the release customer X depends on?" (a chain)
- "What are the main themes across all incident reports?" (global, no single passage)

No single chunk contains the answer, so top-k retrieval returns pieces and the model
guesses the connections. You need the *relationships*, not just the text.

## The idea: model entities and relationships, then traverse

`GraphIndex` builds a property graph:

1. **Extraction** (`_extract`): Gemini reads each parent block and returns
   `(entity, relation, entity)` triples via structured output — e.g.
   `Priya Raman —manages→ Northwind`. Entities become nodes (with a type and the set of
   chunks they appear in); relations become edges (carrying their source chunk for
   provenance). Extraction runs once per parent block to control cost.
2. **Communities** (`_detect_communities`): we cluster the graph (greedy modularity) and
   write a **summary** of each cluster with Gemini. These power global questions.

## Two retrieval modes

### Local search (`local_search`) — relationships around entities

1. Pull the entities out of the question (`_query_entities`).
2. Match them to graph nodes (exact, then substring fallback — `_match_nodes`).
3. Expand **k hops** out (`single_source_shortest_path_length`, `hops=2`).
4. Return the chunks backing the touched nodes/edges, scored by graph distance (closer =
   higher). The result is precise, **auditable** context: you can show the path that
   justified the answer.

This is what answers "who manages the accounts that renewed and opened SSO tickets?" — it
walks `ticket → customer → account manager` instead of hoping one chunk says it all.

### Global search (`global_search`) — big-picture / thematic

Embed the question, compare it to every community summary, and return the top few
summaries. This answers "what are the overall themes?" where the answer lives in the
*structure*, not any single passage.

## Provenance & explainability

Every node stores the chunk ids it came from; every edge stores its source chunk. So a
GraphRAG answer can show **which entities, edges, and passages** supported it — the
explainability that vector-only RAG can't offer, and a big reason graphs are valued in
legal/compliance/support settings.

## This implementation vs production

This is a pragmatic, dependency-light GraphRAG: an in-memory `networkx` graph persisted to
JSON. For large or shared corpora you'd store it in **Neo4j** and traverse with **Cypher**
(`MATCH (c:Customer)-[:OPENED]->(t:Ticket)...`). The retrieval *shape* — match entities,
bound the traversal by hops/types/time, summarize communities for global questions — is
identical. Swapping the backend doesn't change the rest of the pipeline.

## When to bother

Graphs shine when your content is rich in **entities and relationships** (customers,
tickets, people, products, citations) and users ask **connective** questions. If your
traffic is mostly single-passage fact lookups, plain hybrid retrieval may be enough — add
the graph when the eval set shows multi-hop questions failing.
