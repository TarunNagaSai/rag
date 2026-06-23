"""GraphRAG — retrieve *connected context*, not just similar text.

Vector search is great for local lookups but weak on global, multi-hop, cross-document
questions ("which customers renewed AND filed an SSO ticket?"). A knowledge graph
fixes this by modeling entities and the relationships between them.

This is a pragmatic, dependency-light GraphRAG:
  * Extraction  : Gemini pulls (entity, relation, entity) triples per parent block.
  * Graph       : networkx property graph; nodes carry the chunk ids they appear in.
  * Local search: match query entities -> expand k hops -> return the chunks that
                  back those nodes/edges (precise, auditable paths).
  * Global search: detect communities, summarize each, answer big-picture questions
                  from the most relevant community summaries.

For huge corpora you'd persist this in Neo4j and traverse with Cypher; the retrieval
*shape* is identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .gemini import Gemini
from .schema import Chunk, Scored


# --------------------------------------------------------------- extraction schema
class _Entity(BaseModel):
    name: str = Field(description="Canonical entity name (person, org, product, concept, id)")
    type: str = Field(description="Entity type, e.g. PERSON, ORG, PRODUCT, EVENT, CONCEPT")


class _Relation(BaseModel):
    source: str = Field(description="Source entity name")
    relation: str = Field(description="Short relation phrase, e.g. 'renewed', 'owns', 'reported'")
    target: str = Field(description="Target entity name")


class _Extraction(BaseModel):
    entities: list[_Entity] = Field(default_factory=list)
    relations: list[_Relation] = Field(default_factory=list)


class _QueryEntities(BaseModel):
    entities: list[str] = Field(default_factory=list)


def _norm(name: str) -> str:
    return name.strip().lower()


class GraphIndex:
    def __init__(self, settings: Settings | None = None, gemini: Gemini | None = None):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)
        self.graph = nx.MultiDiGraph()
        self.chunks: dict[str, Chunk] = {}        # parent_id -> representative chunk
        self.communities: list[dict] = []         # [{members, summary}]

    # ----------------------------------------------------------------- build
    def build(self, chunks: list[Chunk]) -> "GraphIndex":
        # Extract once per parent block to limit cost; children share a parent.
        parents: dict[str, Chunk] = {}
        for c in chunks:
            parents.setdefault(c.parent_id, c)
        self.chunks.update(parents)

        for pid, chunk in parents.items():
            ext = self._extract(chunk.parent_text or chunk.text)
            for e in ext.entities:
                key = _norm(e.name)
                if not key:
                    continue
                if self.graph.has_node(key):
                    self.graph.nodes[key]["chunks"].add(pid)
                else:
                    self.graph.add_node(key, name=e.name, type=e.type, chunks={pid})
            for r in ext.relations:
                sk, tk = _norm(r.source), _norm(r.target)
                if sk and tk and self.graph.has_node(sk) and self.graph.has_node(tk):
                    self.graph.add_edge(sk, tk, relation=r.relation, chunk=pid)
        self._detect_communities()
        return self

    def _extract(self, text: str) -> _Extraction:
        try:
            return self.g.generate_structured(
                prompt=(
                    "Extract a knowledge graph from the text. Identify the key entities "
                    "and the explicit relationships between them. Use canonical names and "
                    "keep relations short verb phrases. Only extract what the text states.\n\n"
                    f"TEXT:\n{text}"
                ),
                schema=_Extraction,
                system="You are an information-extraction engine that builds knowledge graphs.",
                temperature=0.0,
            )
        except Exception:
            return _Extraction()

    def _detect_communities(self) -> None:
        if self.graph.number_of_nodes() == 0:
            return
        undirected = self.graph.to_undirected()
        try:
            comms = nx.community.greedy_modularity_communities(undirected)
        except Exception:
            comms = [set(undirected.nodes())]
        self.communities = []
        for members in comms:
            members = list(members)
            if not members:
                continue
            summary = self._summarize_community(members)
            self.communities.append({"members": members, "summary": summary})

    def _summarize_community(self, members: list[str]) -> str:
        facts = []
        for u, v, data in self.graph.edges(data=True):
            if u in members or v in members:
                nu = self.graph.nodes[u].get("name", u)
                nv = self.graph.nodes[v].get("name", v)
                facts.append(f"{nu} —{data.get('relation','related to')}→ {nv}")
        if not facts:
            names = ", ".join(self.graph.nodes[m].get("name", m) for m in members[:20])
            return f"Cluster of related entities: {names}"
        fact_block = "\n".join(facts[:60])
        return self.g.generate(
            prompt=(
                "Summarize this cluster of related entities and relationships into a "
                "concise paragraph capturing the big-picture themes.\n\n" + fact_block
            ),
            system="You write query-focused community summaries for GraphRAG.",
            temperature=0.2,
            max_output_tokens=256,
        )

    # ---------------------------------------------------------------- search
    def _query_entities(self, question: str) -> list[str]:
        try:
            out = self.g.generate_structured(
                prompt=f"List the key entities mentioned or implied in this question.\n\n{question}",
                schema=_QueryEntities,
                system="You extract entities to look up in a knowledge graph.",
                temperature=0.0,
            )
            return [_norm(e) for e in out.entities if e.strip()]
        except Exception:
            return []

    def _match_nodes(self, entities: list[str]) -> list[str]:
        matched: list[str] = []
        for e in entities:
            if self.graph.has_node(e):
                matched.append(e)
                continue
            # substring fallback for fuzzy matches
            for node in self.graph.nodes():
                if e in node or node in e:
                    matched.append(node)
        return list(dict.fromkeys(matched))

    def local_search(self, question: str, hops: int = 2,
                     max_chunks: int = 8) -> list[Scored]:
        """Expand from query entities k hops; return the backing chunks with a
        score that decays by graph distance."""
        if self.graph.number_of_nodes() == 0:
            return []
        seeds = self._match_nodes(self._query_entities(question))
        if not seeds:
            return []
        undirected = self.graph.to_undirected()
        chunk_score: dict[str, float] = {}
        for seed in seeds:
            if seed not in undirected:
                continue
            lengths = nx.single_source_shortest_path_length(undirected, seed, cutoff=hops)
            for node, dist in lengths.items():
                weight = 1.0 / (1 + dist)
                for pid in self.graph.nodes[node].get("chunks", set()):
                    chunk_score[pid] = max(chunk_score.get(pid, 0.0), weight)
        ranked = sorted(chunk_score.items(), key=lambda kv: -kv[1])[:max_chunks]
        return [
            Scored(chunk=self.chunks[pid], score=score, how="graph")
            for pid, score in ranked if pid in self.chunks
        ]

    def global_search(self, question: str, top_communities: int = 3) -> list[str]:
        """Return the most relevant community summaries for big-picture questions."""
        if not self.communities:
            return []
        qvec = self.g.embed_query(question)
        sums = [c["summary"] for c in self.communities]
        svecs = self.g.embed(sums)  # doc task type ok for summaries
        sims = svecs @ qvec
        order = sims.argsort()[::-1][:top_communities]
        return [sums[i] for i in order]

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path | None = None) -> Path:
        d = Path(path) if path else self.s.index_dir
        d.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph, edges="links")
        # sets aren't JSON-serializable
        for node in data["nodes"]:
            if isinstance(node.get("chunks"), set):
                node["chunks"] = list(node["chunks"])
        (d / "graph.json").write_text(json.dumps(data, ensure_ascii=False))
        (d / "communities.json").write_text(json.dumps(self.communities, ensure_ascii=False))
        return d

    @classmethod
    def load(cls, chunks: list[Chunk], path: str | Path | None = None,
             settings: Settings | None = None, gemini: Gemini | None = None) -> "GraphIndex":
        s = settings or get_settings()
        d = Path(path) if path else s.index_dir
        gi = cls(s, gemini)
        if not (d / "graph.json").exists():
            return gi
        data = json.loads((d / "graph.json").read_text())
        for node in data["nodes"]:
            if isinstance(node.get("chunks"), list):
                node["chunks"] = set(node["chunks"])
        gi.graph = nx.node_link_graph(data, multigraph=True, directed=True, edges="links")
        if (d / "communities.json").exists():
            gi.communities = json.loads((d / "communities.json").read_text())
        parents: dict[str, Chunk] = {}
        for c in chunks:
            parents.setdefault(c.parent_id, c)
        gi.chunks = parents
        return gi

    @staticmethod
    def exists(path: str | Path | None = None, settings: Settings | None = None) -> bool:
        s = settings or get_settings()
        d = Path(path) if path else s.index_dir
        return (d / "graph.json").exists()
