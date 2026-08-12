

from __future__ import annotations

import heapq
import math
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import httpx
import numpy as np
from fastapi import HTTPException, Query, Request
from pydantic import BaseModel, Field

from httpLib import create_server, run_server

# =====================================================================
# CONFIGURATION  (all overridable via environment variables)
# =====================================================================

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.2")

DIMS = int(os.getenv("DEMO_DIMS", "16"))  # dimensionality of the 20 demo vectors
# Document embeddings' dimensionality is determined at runtime from
# whatever Ollama's embedding model actually outputs (768 for
# nomic-embed-text), exactly like the original C++ DocumentDB.

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "250"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))


# =====================================================================
# DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    """One stored vector: an id, free-text metadata, a category label
    (used by the demo dataset) and the embedding itself."""

    id: int
    metadata: str
    category: str
    embedding: list[float]


@dataclass
class DocItem:
    """One chunk of a user-inserted document, with its real
    Ollama-generated embedding (usually 768 dimensions)."""

    id: int
    title: str
    text: str
    embedding: list[float]


DistFn = Callable[[Sequence[float], Sequence[float]], float]


# =====================================================================
# DISTANCE METRICS
#
# NumPy is used here for the actual vector math (this is the
# "mathematical operations" NumPy is genuinely good at); the surrounding
# algorithms below (Brute Force / KD-Tree / HNSW) are still hand-written
# Python logic, not calls into an existing vector-search library.
# =====================================================================

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Straight-line distance between two points."""
    av, bv = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.sum((av - bv) ** 2)))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine similarity. 0 = identical direction, 1 = orthogonal,
    2 = opposite. Used for text embeddings, where magnitude doesn't
    matter but direction (meaning) does."""
    av, bv = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(av, bv) / (na * nb))


def manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    """Sum of absolute differences along each axis ("city block" distance)."""
    av, bv = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.sum(np.abs(av - bv)))


_METRICS: dict[str, DistFn] = {"cosine": cosine, "manhattan": manhattan, "euclidean": euclidean}


def get_dist_fn(metric: str) -> DistFn:
    return _METRICS.get(metric, euclidean)


# =====================================================================
# BRUTE FORCE
# =====================================================================

class BruteForce:
    """Exact nearest-neighbor search: compare the query against every
    stored vector. O(N*d) per query, but always exact — this is the
    ground truth KD-Tree and HNSW are benchmarked against."""

    def __init__(self) -> None:
        self.items: list[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]

    def knn(self, query: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        scored = [(dist(query, v.embedding), v.id) for v in self.items]
        scored.sort(key=lambda p: p[0])
        return scored[:k]


# =====================================================================
# KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree:
    """
    Binary space-partitioning tree. Each node splits space along one
    dimension, cycling through all dimensions as you go deeper. Search
    prunes an entire subtree whenever the closest possible point in it
    can't beat the current worst kept candidate (the classic
    "ball-within-hyperslab" check).

    Degrades toward brute-force performance in very high dimensions
    (the curse of dimensionality) — fine for this project's 16D demo
    vectors, less useful for the 768D document embeddings, which is why
    DocumentDB below sticks to HNSW + Brute Force only, matching the
    original C++ design.
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: Optional[KDNode] = None

    def insert(self, item: VectorItem) -> None:
        self.root = self._insert(self.root, item, 0)

    def _insert(self, node: Optional[KDNode], item: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(item)
        axis = depth % self.dims
        if item.embedding[axis] < node.item.embedding[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def rebuild(self, items: list[VectorItem]) -> None:
        """KD-Trees don't support efficient in-place deletion, so after
        a removal we rebuild from the surviving items — same as the
        original C++ implementation."""
        self.root = None
        for item in items:
            self.insert(item)

    def knn(self, query: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        heap: list[tuple[float, int]] = []  # (-distance, id) max-heap
        self._knn(self.root, query, k, 0, dist, heap)
        result = [(-nd, i) for nd, i in heap]
        result.sort(key=lambda p: p[0])
        return result

    def _knn(self, node, query, k, depth, dist, heap) -> None:
        if node is None:
            return
        d = dist(query, node.item.embedding)
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.item.id))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.item.id))

        axis = depth % self.dims
        diff = query[axis] - node.item.embedding[axis]
        closer, farther = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._knn(closer, query, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, query, k, depth + 1, dist, heap)


# =====================================================================
# HNSW — Hierarchical Navigable Small World
#
# Same family of algorithm used internally by Pinecone, Weaviate,
# Chroma and Milvus. Vectors are inserted into a multilayer graph:
# every node is randomly assigned a "max layer" (fewer nodes exist as
# you go up), and each node connects to its M nearest neighbors on
# every layer it belongs to. Search starts at a single entry point on
# the top layer and greedily descends — like taking a highway to the
# right neighborhood, then local streets (layer 0) to the exact spot.
# =====================================================================

class HNSWNode:
    __slots__ = ("item", "max_layer", "neighbors")

    def __init__(self, item: VectorItem, max_layer: int) -> None:
        self.item = item
        self.max_layer = max_layer
        # neighbors[layer] = list of neighbor node ids on that layer
        self.neighbors: list[list[int]] = [[] for _ in range(max_layer + 1)]


class HNSWIndex:
    def __init__(self, m: int = 16, ef_construction: int = 200, seed: int = 42) -> None:
        self.m = m
        self.m0 = 2 * m  # layer 0 keeps more connections since it holds every node
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(m)  # controls how quickly layers thin out
        self._rng = random.Random(seed)

        self.graph: dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_point: Optional[int] = None

    def _random_level(self) -> int:
        u = max(self._rng.random(), 1e-12)
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(
        self, query: list[float], entry_id: int, ef: int, layer: int, dist: DistFn
    ) -> list[tuple[float, int]]:
        """Greedy beam search on a single layer: expand outward from
        ``entry_id`` until the closest unexplored candidate can no
        longer beat the worst of the ``ef`` best results found so far."""
        visited = {entry_id}
        d0 = dist(query, self.graph[entry_id].item.embedding)
        candidates: list[tuple[float, int]] = [(d0, entry_id)]  # min-heap
        found: list[tuple[float, int]] = [(-d0, entry_id)]  # max-heap (negated)

        while candidates:
            cur_dist, cur_id = heapq.heappop(candidates)
            if len(found) >= ef and cur_dist > -found[0][0]:
                break
            node = self.graph.get(cur_id)
            if node is None or layer >= len(node.neighbors):
                continue
            for nbr_id in node.neighbors[layer]:
                if nbr_id in visited or nbr_id not in self.graph:
                    continue
                visited.add(nbr_id)
                nd = dist(query, self.graph[nbr_id].item.embedding)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(candidates, (nd, nbr_id))
                    heapq.heappush(found, (-nd, nbr_id))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-nd, i) for nd, i in found]
        result.sort(key=lambda p: p[0])
        return result

    @staticmethod
    def _select_neighbors(candidates: list[tuple[float, int]], max_m: int) -> list[int]:
        """Keep the closest max_m candidates as this node's neighbors."""
        return [node_id for _, node_id in candidates[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        node_id = item.id
        level = self._random_level()
        self.graph[node_id] = HNSWNode(item, level)

        if self.entry_point is None:
            self.entry_point = node_id
            self.top_layer = level
            return

        entry = self.entry_point

        # Phase 1: descend from the top layer to `level + 1`, moving
        # greedily to the single closest node at each layer.
        for layer in range(self.top_layer, level, -1):
            if layer < len(self.graph[entry].neighbors):
                nearest = self._search_layer(item.embedding, entry, 1, layer, dist)
                if nearest:
                    entry = nearest[0][1]

        # Phase 2: from min(top_layer, level) down to 0, run a real beam
        # search and wire up bidirectional connections.
        for layer in range(min(self.top_layer, level), -1, -1):
            candidates = self._search_layer(item.embedding, entry, self.ef_construction, layer, dist)
            max_m = self.m0 if layer == 0 else self.m
            selected = self._select_neighbors(candidates, max_m)
            self.graph[node_id].neighbors[layer] = selected

            for nbr_id in selected:
                nbr = self.graph.get(nbr_id)
                if nbr is None:
                    continue
                if len(nbr.neighbors) <= layer:
                    nbr.neighbors.extend([] for _ in range(layer + 1 - len(nbr.neighbors)))
                conn = nbr.neighbors[layer]
                conn.append(node_id)
                if len(conn) > max_m:
                    scored = sorted(
                        ((dist(nbr.item.embedding, self.graph[c].item.embedding), c)
                         for c in conn if c in self.graph),
                        key=lambda p: p[0],
                    )
                    nbr.neighbors[layer] = [c for _, c in scored[:max_m]]

            if candidates:
                entry = candidates[0][1]

        if level > self.top_layer:
            self.top_layer = level
            self.entry_point = node_id

    def knn(self, query: list[float], k: int, ef: int, dist: DistFn) -> list[tuple[float, int]]:
        if self.entry_point is None:
            return []
        entry = self.entry_point
        for layer in range(self.top_layer, 0, -1):
            if layer < len(self.graph[entry].neighbors):
                nearest = self._search_layer(query, entry, 1, layer, dist)
                if nearest:
                    entry = nearest[0][1]
        result = self._search_layer(query, entry, max(ef, k), 0, dist)
        return result[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self.graph:
            return
        for node in self.graph.values():
            for layer_neighbors in node.neighbors:
                if item_id in layer_neighbors:
                    layer_neighbors.remove(item_id)
        if self.entry_point == item_id:
            self.entry_point = next((nid for nid in self.graph if nid != item_id), None)
        del self.graph[item_id]

    def get_info(self) -> dict:
        """Snapshot of the graph structure for the /hnsw-info endpoint,
        which the frontend uses to visualize layer sizes."""
        max_layers = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_layers
        edges_per_layer = [0] * max_layers
        nodes, edges = [], []

        for node_id, node in self.graph.items():
            nodes.append({
                "id": node_id, "metadata": node.item.metadata,
                "category": node.item.category, "maxLyr": node.max_layer,
            })
            for layer in range(min(node.max_layer, max_layers - 1) + 1):
                nodes_per_layer[layer] += 1
                if layer < len(node.neighbors):
                    for nbr_id in node.neighbors[layer]:
                        if node_id < nbr_id:  # count each undirected edge once
                            edges_per_layer[layer] += 1
                            edges.append({"src": node_id, "dst": nbr_id, "lyr": layer})

        return {
            "topLayer": self.top_layer, "nodeCount": len(self.graph),
            "nodesPerLayer": nodes_per_layer, "edgesPerLayer": edges_per_layer,
            "nodes": nodes, "edges": edges,
        }

    def __len__(self) -> int:
        return len(self.graph)


# =====================================================================
# VECTOR DATABASE  (demo 16D index — mirrors the original VectorDB class)
# =====================================================================

class VectorDB:
    """Unified interface over Brute Force, KD-Tree and HNSW. A single
    insert/delete keeps all three indexes in sync so search results and
    benchmarks stay directly comparable."""

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._store: dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSWIndex(16, 200)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, embedding: list[float], dist: DistFn) -> int:
        with self._lock:
            item = VectorItem(self._next_id, metadata, category, embedding)
            self._next_id += 1
            self._store[item.id] = item
            self._bf.insert(item)
            self._kdt.insert(item)
            self._hnsw.insert(item, dist)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bf.remove(item_id)
            self._hnsw.remove(item_id)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, query: list[float], k: int, metric: str, algo: str) -> dict:
        with self._lock:
            dist = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self._bf.knn(query, k, dist)
            elif algo == "kdtree":
                raw = self._kdt.knn(query, k, dist)
            else:
                raw = self._hnsw.knn(query, k, 50, dist)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = [
                {"id": i, "metadata": self._store[i].metadata, "category": self._store[i].category,
                 "distance": d, "embedding": self._store[i].embedding}
                for d, i in raw if i in self._store
            ]
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, query: list[float], k: int, metric: str) -> dict:
        with self._lock:
            dist = get_dist_fn(metric)

            def timed(fn) -> int:
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            bf_us = timed(lambda: self._bf.knn(query, k, dist))
            kd_us = timed(lambda: self._kdt.knn(query, k, dist))
            hnsw_us = timed(lambda: self._hnsw.knn(query, k, 50, dist))
            return {"bruteforceUs": bf_us, "kdtreeUs": kd_us, "hnswUs": hnsw_us, "itemCount": len(self._store)}

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# =====================================================================
# TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    """Split long text into overlapping word-based chunks so each chunk
    can be embedded and retrieved independently. Short documents are
    returned unchanged as a single chunk."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks: list[str] = []
    step = max(chunk_words - overlap_words, 1)
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks


# =====================================================================
# OLLAMA CLIENT — wraps the local Ollama REST API
# Install: https://ollama.com
# Models:  ollama pull nomic-embed-text
#          ollama pull llama3.2
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434,
                 embed_model: str = "nomic-embed-text", gen_model: str = "llama3.2") -> None:
        self.base_url = f"http://{host}:{port}"
        self.embed_model = embed_model
        self.gen_model = gen_model

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def embed(self, text: str) -> list[float]:
        """Returns [] if Ollama is unreachable or the model isn't installed."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                )
                if r.status_code != 200:
                    return []
                return list(r.json().get("embedding", []))
        except httpx.HTTPError:
            return []

    async def generate(self, prompt: str) -> str:
        """LLMs can be slow on CPU — generous 180s timeout, matching the original."""
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                r = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.gen_model, "prompt": prompt, "stream": False},
                )
                if r.status_code != 200:
                    return "ERROR: Ollama unavailable. Run: ollama serve"
                return str(r.json().get("response", ""))
        except httpx.HTTPError:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
# DOCUMENT DATABASE — HNSW over real Ollama embeddings
# =====================================================================

class DocumentDB:
    """Stores document chunks with their real (usually 768D) Ollama
    embeddings. Falls back to Brute Force for small collections, since
    HNSW's approximate search isn't worth it until there's enough data —
    same threshold the original C++ DocumentDB used."""

    _BRUTE_FORCE_THRESHOLD = 10

    def __init__(self) -> None:
        self._store: dict[int, DocItem] = {}
        self._hnsw = HNSWIndex(16, 200)
        self._bf = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims = 0

    def insert(self, title: str, text: str, embedding: list[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(embedding)
            item = DocItem(self._next_id, title, text, embedding)
            self._next_id += 1
            self._store[item.id] = item

            vi = VectorItem(item.id, title, "doc", embedding)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, query: list[float], k: int, max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < self._BRUTE_FORCE_THRESHOLD:
                raw = self._bf.knn(query, k, cosine)
            else:
                raw = self._hnsw.knn(query, k, 50, cosine)
            return [(d, self._store[i]) for d, i in raw if i in self._store and d <= max_dist]

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._hnsw.remove(item_id)
            self._bf.remove(item_id)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self._store.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def dims(self) -> int:
        return self._dims


# =====================================================================
# DEMO DATA  (20 hand-crafted 16D vectors — CS / Math / Food / Sports)
# Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
# =====================================================================

_DEMO_ITEMS: list[tuple[str, str, list[float]]] = [
    ("Linked List: nodes connected by pointers", "cs",
     [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
    ("Calculus: derivatives integrals and limits", "math",
     [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
]


def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    for metadata, category, embedding in _DEMO_ITEMS:
        db.insert(metadata, category, embedding, dist)


# =====================================================================
# REQUEST / RESPONSE MODELS  (Pydantic — used for validation only)
# =====================================================================

class InsertRequest(BaseModel):
    metadata: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    embedding: list[float]


class DocInsertRequest(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class DocAskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = 3


class DocSearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = 3


# =====================================================================
# HTTP SERVER  (built on httpLib.py — Python's equivalent of httplib.h)
# =====================================================================

app = create_server("My Own AI")

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient(OLLAMA_HOST, OLLAMA_PORT, EMBED_MODEL, GEN_MODEL)


@asynccontextmanager
async def _lifespan(_app):
    load_demo(db)
    ollama_up = await ollama.is_available()
    print("=== My Own AI — Vector Database Engine ===")
    print(f"http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"{len(db)} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {EMBED_MODEL}  gen model: {GEN_MODEL}")
    yield


app.router.lifespan_context = _lifespan


# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

@app.get("/search")
def search(v: str = Query(...), k: int = Query(5, ge=1, le=100),
           metric: str = Query("cosine"), algo: str = Query("hnsw")) -> dict:
    try:
        query = [float(x) for x in v.split(",") if x.strip() != ""]
    except ValueError:
        raise HTTPException(400, "invalid vector")
    if len(query) != DIMS:
        raise HTTPException(400, f"need {DIMS}D vector")
    if algo not in ("bruteforce", "kdtree", "hnsw"):
        raise HTTPException(400, "invalid algorithm")
    if metric not in _METRICS:
        raise HTTPException(400, "invalid distance metric")
    return db.search(query, k, metric, algo)


@app.post("/insert")
def insert(body: InsertRequest) -> dict:
    if len(body.embedding) != DIMS:
        raise HTTPException(400, f"need {DIMS}D vector")
    item_id = db.insert(body.metadata, body.category, body.embedding, get_dist_fn("cosine"))
    return {"id": item_id}


@app.delete("/delete/{item_id}")
def delete(item_id: int) -> dict:
    return {"ok": db.remove(item_id)}


@app.get("/items")
def items() -> list[dict]:
    return [
        {"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.embedding}
        for v in db.all()
    ]


@app.get("/benchmark")
def benchmark(v: str = Query(...), k: int = Query(5, ge=1, le=100), metric: str = Query("cosine")) -> dict:
    try:
        query = [float(x) for x in v.split(",") if x.strip() != ""]
    except ValueError:
        raise HTTPException(400, "invalid vector")
    if len(query) != DIMS:
        raise HTTPException(400, f"need {DIMS}D vector")
    return db.benchmark(query, k, metric)


@app.get("/hnsw-info")
def hnsw_info() -> dict:
    return db.hnsw_info()


@app.get("/stats")
def stats() -> dict:
    return {
        "count": len(db), "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    }


# ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

_OLLAMA_UNAVAILABLE_MSG = (
    "Ollama unavailable. Install from https://ollama.com then run: "
    "ollama pull nomic-embed-text && ollama pull llama3.2"
)


@app.post("/doc/insert")
async def doc_insert(body: DocInsertRequest) -> dict:
    chunks = chunk_text(body.text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)
    if not chunks:
        raise HTTPException(400, "need title and text")

    ids: list[int] = []
    for i, chunk in enumerate(chunks):
        embedding = await ollama.embed(chunk)
        if not embedding:
            raise HTTPException(503, _OLLAMA_UNAVAILABLE_MSG)
        chunk_title = f"{body.title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else body.title
        ids.append(doc_db.insert(chunk_title, chunk, embedding))

    return {"ids": ids, "chunks": len(chunks), "dims": doc_db.dims}


@app.delete("/doc/delete/{item_id}")
def doc_delete(item_id: int) -> dict:
    return {"ok": doc_db.remove(item_id)}


@app.get("/doc/list")
def doc_list() -> list[dict]:
    out = []
    for doc in doc_db.all():
        preview = doc.text[:120] + ("…" if len(doc.text) > 120 else "")
        out.append({"id": doc.id, "title": doc.title, "preview": preview, "words": doc.text.count(" ") + 1})
    return out


@app.post("/doc/search")
async def doc_search(body: DocSearchRequest) -> dict:
    """Fast retrieval-only endpoint used by the frontend's scatter-plot
    visualizer while the full RAG answer is still generating."""
    q_emb = await ollama.embed(body.question)
    if not q_emb:
        raise HTTPException(503, "Ollama unavailable")
    hits = doc_db.search(q_emb, body.k)
    return {"contexts": [{"id": doc.id, "title": doc.title, "distance": d} for d, doc in hits]}


@app.post("/doc/ask")
async def doc_ask(body: DocAskRequest) -> dict:
    """
    Full RAG pipeline:
      1. Embed the question
      2. Retrieve the top-k most relevant chunks
      3. Build a context block from those chunks
      4. Send context + question to the local LLM
      5. Return the generated answer along with the chunks used
    """
    q_emb = await ollama.embed(body.question)
    if not q_emb:
        raise HTTPException(503, "Ollama unavailable")

    hits = doc_db.search(q_emb, body.k)
    ctx = "".join(f"[{i + 1}] {doc.title}:\n{doc.text}\n\n" for i, (_, doc) in enumerate(hits))
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}"
        f"Question: {body.question}\n\n"
        "Answer:"
    )
    answer = await ollama.generate(prompt)

    return {
        "answer": answer, "model": ollama.gen_model,
        "contexts": [{"id": doc.id, "title": doc.title, "text": doc.text, "distance": d} for d, doc in hits],
        "docCount": len(doc_db),
    }


@app.get("/status")
async def status() -> dict:
    return {
        "ollamaAvailable": await ollama.is_available(),
        "embedModel": ollama.embed_model, "genModel": ollama.gen_model,
        "docCount": len(doc_db), "docDims": doc_db.dims,
        "demoDims": DIMS, "demoCount": len(db),
    }


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    run_server(app, SERVER_HOST, SERVER_PORT)
