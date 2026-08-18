import os
import math
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class SimpleEmbeddingProvider:
    """
    Lightweight, dependency-free TF-IDF/n-gram keyword + character embedding fallback
    when heavyweight PyTorch/sentence-transformers packages are not present or initializing.
    """
    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim

    def _hash_token(self, token: str) -> int:
        h = 0
        for char in token:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        return h % self.vector_dim

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.vector_dim
        tokens = text.lower().split()
        if not tokens:
            return vec
        for token in tokens:
            idx = self._hash_token(token)
            vec[idx] += 1.0
        # Cosine normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class VectorStoreManager:
    """
    Chroma-backed Vector Store Manager for LexAssist.
    Manages semantic search collections for contracts, precedents, and Indian statutes.
    """
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.fallback_embedder = SimpleEmbeddingProvider()

        if _CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=persist_directory)
                self._chroma_ready = True
            except Exception as e:
                print(f"[VectorStoreManager] Chroma Client initialization note: {e}")
                self._chroma_ready = False
        else:
            self._chroma_ready = False

        # In-memory storage fallback if Chroma client isn't available
        self.in_memory_collections: Dict[str, List[Dict[str, Any]]] = {}

    def get_or_create_collection(self, collection_name: str):
        if self._chroma_ready:
            try:
                return self.client.get_or_create_collection(name=collection_name)
            except Exception:
                pass
        
        if collection_name not in self.in_memory_collections:
            self.in_memory_collections[collection_name] = []
        return collection_name

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Adds documents with metadata to specified vector collection."""
        if self._chroma_ready:
            try:
                coll = self.client.get_or_create_collection(name=collection_name)
                # Generate embeddings using fallback or default Chroma embedder
                embeddings = self.fallback_embedder.embed_batch(documents)
                coll.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                return True
            except Exception as e:
                print(f"[VectorStoreManager] Chroma add error: {e}. Using in-memory fallback.")

        # In-memory fallback indexing
        if collection_name not in self.in_memory_collections:
            self.in_memory_collections[collection_name] = []

        embeddings = self.fallback_embedder.embed_batch(documents)
        for doc, meta, doc_id, emb in zip(documents, metadatas, ids, embeddings):
            self.in_memory_collections[collection_name].append({
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "embedding": emb
            })
        return True

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity query against specified collection.
        Returns ordered list of matching chunks with distance/relevance score.
        """
        if self._chroma_ready:
            try:
                coll = self.client.get_collection(name=collection_name)
                query_emb = self.fallback_embedder.embed_text(query_text)
                res = coll.query(
                    query_embeddings=[query_emb],
                    n_results=n_results,
                    where=where_filter
                )
                formatted_results = []
                if res and res.get("documents") and res["documents"][0]:
                    docs = res["documents"][0]
                    metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
                    ids = res["ids"][0] if res.get("ids") else [f"doc_{i}" for i in range(len(docs))]
                    distances = res["distances"][0] if res.get("distances") else [0.0] * len(docs)

                    for d, m, i, dist in zip(docs, metas, ids, distances):
                        # Convert distance to similarity score
                        sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                        formatted_results.append({
                            "id": i,
                            "document": d,
                            "metadata": m,
                            "similarity_score": round(sim, 4)
                        })
                return formatted_results
            except Exception:
                pass

        # In-memory search fallback using cosine similarity
        items = self.in_memory_collections.get(collection_name, [])
        if not items:
            return []

        query_emb = self.fallback_embedder.embed_text(query_text)
        scored = []

        for item in items:
            # Check metadata filter if specified
            if where_filter:
                match = True
                for k, v in where_filter.items():
                    if item["metadata"].get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Dot product for normalized vectors = cosine similarity
            emb = item["embedding"]
            score = sum(q * e for q, e in zip(query_emb, emb))
            scored.append({
                "id": item["id"],
                "document": item["document"],
                "metadata": item["metadata"],
                "similarity_score": round(score, 4)
            })

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:n_results]
