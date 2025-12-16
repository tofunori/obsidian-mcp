"""
Hybrid Retriever simplifie pour Obsidian.
Base sur ragdoc-mcp/src/hybrid_retriever.py mais sans chunking.
"""

import logging
import threading
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
import chromadb

logger = logging.getLogger(__name__)


class ObsidianRetriever:
    """
    Recherche hybride BM25 + Semantique pour notes Obsidian.

    Simplifie par rapport a ragdoc:
    - Pas de chunking (notes entieres)
    - Tokenisation simple (pas de stemming scientifique)
    - Filtrage par tags et dossiers
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedding_function=None
    ):
        """
        Args:
            collection: Collection ChromaDB
            embedding_function: Fonction pour embedder les requetes
        """
        self.collection = collection
        self.embedding_function = embedding_function

        # Index BM25 (lazy init)
        self.docs: List[str] = []
        self.ids: List[str] = []
        self.metadatas: List[dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self._id_to_idx: dict[str, int] = {}
        self._bm25_lock = threading.Lock()
        self._bm25_building = False

    def _tokenize(self, text: str) -> List[str]:
        """Tokenisation simple pour notes personnelles."""
        # Lowercase et split sur espaces/ponctuation
        text = text.lower()
        # Garder les mots alphanumeriques
        tokens = []
        current = []
        for char in text:
            if char.isalnum():
                current.append(char)
            else:
                if current:
                    tokens.append(''.join(current))
                    current = []
        if current:
            tokens.append(''.join(current))

        # Filtrer les mots trop courts
        return [t for t in tokens if len(t) > 2]

    def _build_bm25_index(self):
        """Construit l'index BM25 depuis ChromaDB."""
        all_data = self.collection.get(include=["documents", "metadatas"])

        self.docs = all_data['documents'] or []
        self.ids = all_data['ids'] or []
        self.metadatas = all_data['metadatas'] or []
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}

        if not self.docs:
            logger.warning("Aucun document dans la collection")
            return

        # Tokeniser le corpus
        tokenized_corpus = [self._tokenize(doc) for doc in self.docs]

        # Creer l'index BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Index BM25 construit: {len(self.docs)} notes")

    def ensure_bm25_index(self, background: bool = True) -> bool:
        """
        S'assure que l'index BM25 est pret.

        Returns:
            True si pret, False si en construction
        """
        if self.bm25 is not None:
            return True

        with self._bm25_lock:
            if self.bm25 is not None:
                return True
            if self._bm25_building:
                return False
            self._bm25_building = True

        def build_worker():
            try:
                self._build_bm25_index()
            finally:
                with self._bm25_lock:
                    self._bm25_building = False

        if background:
            t = threading.Thread(target=build_worker, daemon=True)
            t.start()
            return False

        build_worker()
        return self.bm25 is not None

    def rebuild_index(self):
        """Force la reconstruction de l'index BM25."""
        self.bm25 = None
        self.ensure_bm25_index(background=False)

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7,
        folder: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Recherche hybride.

        Args:
            query: Requete de recherche
            top_k: Nombre de resultats
            alpha: Poids semantique (0.7 = 70% semantic, 30% BM25)
            folder: Filtrer par dossier
            tags: Filtrer par tags

        Returns:
            Liste de dicts avec: id, text, metadata, score
        """
        # Construire le filtre ChromaDB
        where = self._build_where_filter(folder, tags)

        # BM25 search
        bm25_results = []
        if alpha < 1.0:
            self.ensure_bm25_index(background=False)
            if self.bm25 is not None:
                bm25_results = self._bm25_search(query, top_n=100, where=where)

        # Semantic search
        semantic_results, payload = self._semantic_search(query, top_n=100, where=where)

        # Fusion RRF
        fused = self._reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=alpha, payload=payload
        )

        return fused[:top_k]

    def find_similar(self, note_path: str, top_k: int = 5) -> List[Dict]:
        """
        Trouve les notes similaires a une note donnee.

        Args:
            note_path: Chemin de la note source
            top_k: Nombre de resultats

        Returns:
            Notes similaires (exclut la note source)
        """
        # Recuperer l'embedding de la note
        try:
            result = self.collection.get(
                where={"path": note_path},
                include=["embeddings", "documents", "metadatas"]
            )
            if not result['embeddings'] or not result['embeddings'][0]:
                return []

            note_embedding = result['embeddings'][0]
        except Exception:
            return []

        # Chercher les notes similaires
        similar = self.collection.query(
            query_embeddings=[note_embedding],
            n_results=top_k + 1,  # +1 pour exclure la note source
            include=["documents", "metadatas", "distances"]
        )

        results = []
        for i, doc_id in enumerate(similar['ids'][0]):
            # Exclure la note source
            if similar['metadatas'][0][i].get('path') == note_path:
                continue

            distance = similar['distances'][0][i]
            results.append({
                'id': doc_id,
                'text': similar['documents'][0][i],
                'metadata': similar['metadatas'][0][i],
                'score': 1 - distance,  # Convertir distance en similarite
            })

            if len(results) >= top_k:
                break

        return results

    def _build_where_filter(
        self,
        folder: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[dict]:
        """Construit le filtre ChromaDB."""
        conditions = []

        if folder:
            # Filtre par dossier (prefix match)
            conditions.append({"vault_path": {"$contains": folder}})

        if tags:
            # Filtre par tags (au moins un tag present)
            for tag in tags:
                conditions.append({"tags": {"$contains": tag}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def _bm25_search(
        self,
        query: str,
        top_n: int,
        where: Optional[dict] = None
    ) -> List[Tuple[str, float, int]]:
        """Recherche BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Filtrer si necessaire
        eligible = list(range(len(scores)))
        if where:
            eligible = [
                i for i, meta in enumerate(self.metadatas)
                if self._match_where(meta, where)
            ]

        if not eligible:
            return []

        # Top-N
        scores_arr = np.asarray(scores)[eligible]
        top_indices = np.argsort(scores_arr)[::-1][:top_n]

        results = []
        for rank, local_idx in enumerate(top_indices):
            global_idx = eligible[int(local_idx)]
            doc_id = self.ids[global_idx]
            score = float(scores_arr[int(local_idx)])
            results.append((doc_id, score, rank))

        return results

    def _semantic_search(
        self,
        query: str,
        top_n: int,
        where: Optional[dict] = None
    ) -> Tuple[List[Tuple[str, float, int]], Dict]:
        """Recherche semantique via ChromaDB."""
        if self.embedding_function is None:
            return [], {}

        query_embedding = self.embedding_function([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        if not results['ids'] or not results['ids'][0]:
            return [], {}

        semantic_results = []
        payload = {}

        for rank, (doc_id, distance) in enumerate(zip(
            results['ids'][0], results['distances'][0]
        )):
            similarity = 1 - distance
            semantic_results.append((doc_id, similarity, rank))

            if rank < len(results['documents'][0]):
                payload[doc_id] = (
                    results['documents'][0][rank],
                    results['metadatas'][0][rank]
                )

        return semantic_results, payload

    def _match_where(self, metadata: dict, where: dict) -> bool:
        """Evalue un filtre where simple."""
        if not where or not metadata:
            return True

        if "$and" in where:
            return all(self._match_where(metadata, c) for c in where["$and"])

        if "$or" in where:
            return any(self._match_where(metadata, c) for c in where["$or"])

        for key, condition in where.items():
            value = metadata.get(key, "")

            if isinstance(condition, dict):
                if "$contains" in condition:
                    if condition["$contains"] not in str(value):
                        return False
            else:
                if value != condition:
                    return False

        return True

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float, int]],
        semantic_results: List[Tuple[str, float, int]],
        alpha: float = 0.7,
        k: int = 60,
        payload: Optional[Dict] = None
    ) -> List[Dict]:
        """Fusion RRF des resultats."""
        scores = defaultdict(lambda: {'bm25': 0, 'semantic': 0})

        # BM25 contribution
        for doc_id, raw_score, rank in bm25_results:
            scores[doc_id]['bm25'] = 1 / (k + rank + 1)
            scores[doc_id]['bm25_raw'] = raw_score

        # Semantic contribution
        for doc_id, raw_score, rank in semantic_results:
            scores[doc_id]['semantic'] = 1 / (k + rank + 1)
            scores[doc_id]['semantic_raw'] = raw_score

        # Combined score
        for doc_id in scores:
            scores[doc_id]['combined'] = (
                alpha * scores[doc_id]['semantic'] +
                (1 - alpha) * scores[doc_id]['bm25']
            )

        # Trier et formater
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1]['combined'],
            reverse=True
        )

        final_results = []
        for doc_id, score_data in sorted_results:
            text, metadata = None, None

            if payload and doc_id in payload:
                text, metadata = payload[doc_id]

            if text is None:
                idx = self._id_to_idx.get(doc_id)
                if idx is not None:
                    text = self.docs[idx]
                    metadata = self.metadatas[idx]

            if text is None:
                continue

            final_results.append({
                'id': doc_id,
                'text': text,
                'metadata': metadata or {},
                'score': score_data['combined'],
                'bm25_score': score_data.get('bm25_raw', 0),
                'semantic_score': score_data.get('semantic_raw', 0),
            })

        return final_results
