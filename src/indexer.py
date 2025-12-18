"""
Indexeur pour vault Obsidian.
Gere l'indexation incrementale des notes dans ChromaDB.
"""

import os
import gc
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings
import voyageai

from .note_parser import parse_note, scan_vault
from .wikilink_graph import WikilinkGraph

logger = logging.getLogger(__name__)


class ObsidianIndexer:
    """
    Indexeur de notes Obsidian avec support incremental.

    Features:
    - Indexation complete ou incrementale (MD5 hash)
    - Embeddings via Voyage AI
    - Graphe de wikilinks
    """

    def __init__(
        self,
        vault_path: str,
        db_path: str = "./chroma_db",
        collection_name: str = "obsidian_notes",
        voyage_api_key: Optional[str] = None
    ):
        """
        Args:
            vault_path: Chemin vers le vault Obsidian
            db_path: Chemin vers la base ChromaDB
            collection_name: Nom de la collection
            voyage_api_key: Cle API Voyage (ou env VOYAGE_API_KEY)
        """
        self.vault_path = Path(vault_path)
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        # Voyage AI client
        api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY requis")
        self.voyage = voyageai.Client(api_key=api_key)

        # ChromaDB
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Graphe de liens
        self.graph = WikilinkGraph()
        self.graph_path = self.db_path / "wikilink_graph.json"
        if self.graph_path.exists():
            self.graph.load(str(self.graph_path))

    def embed_texts(self, texts: list[str], model: str = "voyage-3") -> list[list[float]]:
        """
        Genere des embeddings via Voyage AI.

        Pour les notes Obsidian (documents entiers), utilise voyage-3 standard.
        voyage-context-3 necessite du chunking et n'est pas adapte aux documents entiers.
        """
        if not texts:
            return []

        # Filtrer les textes vides et tronquer les textes trop longs
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                # Texte vide - utiliser un placeholder
                processed_texts.append("[Note vide]")
            elif len(text) > 32000:
                # Tronquer les textes trop longs (limite ~8k tokens)
                processed_texts.append(text[:32000])
            else:
                processed_texts.append(text)

        result = self.voyage.embed(
            texts=processed_texts,
            model=model,
            input_type="document"
        )
        return result.embeddings

    def _force_persist(self):
        """
        Force la persistance ChromaDB en recreant le client.

        ChromaDB utilise SQLite avec WAL. Les donnees sont ecrites dans le WAL
        mais pas immediatement checkpoint vers le fichier principal. Un autre
        processus peut ne pas voir les donnees non-checkpoint.

        Cette methode force le checkpoint en fermant et recreant le client.
        """
        collection_name = self.collection_name
        db_path = self.db_path

        # Supprimer les references pour forcer la fermeture
        del self.collection
        del self.chroma
        gc.collect()

        # Recreer le client (force le checkpoint WAL)
        self.chroma = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB persiste (checkpoint WAL force)")

    def get_indexed_hashes(self) -> dict[str, str]:
        """Recupere les hashes des notes deja indexees."""
        try:
            result = self.collection.get(include=["metadatas"])
            hashes = {}
            for i, meta in enumerate(result['metadatas'] or []):
                if meta and 'path' in meta and 'hash' in meta:
                    hashes[meta['path']] = meta['hash']
            return hashes
        except Exception:
            return {}

    def index_vault(self, incremental: bool = True) -> dict:
        """
        Indexe le vault Obsidian.

        Args:
            incremental: Si True, n'indexe que les notes modifiees

        Returns:
            Statistiques d'indexation
        """
        stats = {
            'total_notes': 0,
            'indexed': 0,
            'skipped': 0,
            'deleted': 0,
            'errors': 0,
            'started_at': datetime.now().isoformat()
        }

        # Scanner le vault
        logger.info(f"Scan du vault: {self.vault_path}")
        notes = scan_vault(
            str(self.vault_path),
            exclude_folders=['.obsidian', '.trash', '.git']
        )
        stats['total_notes'] = len(notes)
        logger.info(f"Trouve {len(notes)} notes")

        # Hashes existants pour incremental
        existing_hashes = self.get_indexed_hashes() if incremental else {}

        # Chemins actuels pour detecter les suppressions
        current_paths = {note['path'] for note in notes}

        # Detecter les notes supprimees
        for path in list(existing_hashes.keys()):
            if path not in current_paths:
                try:
                    # Trouver l'ID du document
                    result = self.collection.get(
                        where={"path": path},
                        include=[]
                    )
                    if result['ids']:
                        self.collection.delete(ids=result['ids'])
                        stats['deleted'] += 1
                        logger.info(f"Supprime: {path}")
                except Exception as e:
                    logger.warning(f"Erreur suppression {path}: {e}")

        # Indexer les notes nouvelles/modifiees
        notes_to_index = []
        for note in notes:
            path = note['path']
            current_hash = note['hash']

            if incremental and path in existing_hashes:
                if existing_hashes[path] == current_hash:
                    stats['skipped'] += 1
                    continue

            notes_to_index.append(note)

        if not notes_to_index:
            logger.info("Aucune note a indexer")
        else:
            logger.info(f"Indexation de {len(notes_to_index)} notes...")
            self._index_notes(notes_to_index, stats)

        # Reconstruire le graphe de liens
        logger.info("Reconstruction du graphe de liens...")
        self.graph.rebuild_from_notes(notes)
        self.graph.save(str(self.graph_path))

        # Verification post-indexation
        db_count = self.collection.count()
        expected = stats['total_notes']
        if db_count != expected:
            diff = expected - db_count
            logger.warning(f"Incoherence detectee: {db_count} documents dans ChromaDB, {expected} notes dans le vault (manquantes: {diff})")
            stats['warning'] = f"{diff} notes non indexees - relancer une indexation complete"
        else:
            logger.info(f"Verification OK: {db_count} documents")

        # Forcer la persistance pour que d'autres processus voient les donnees
        self._force_persist()

        stats['finished_at'] = datetime.now().isoformat()
        return stats

    def _index_notes(self, notes: list[dict], stats: dict):
        """Indexe une liste de notes."""
        batch_size = 10

        for i in range(0, len(notes), batch_size):
            batch = notes[i:i + batch_size]

            try:
                # Preparer les textes pour embedding
                texts = [note['content'] for note in batch]

                # Generer les embeddings
                embeddings = self.embed_texts(texts)

                # Preparer les metadonnees
                ids = []
                documents = []
                metadatas = []

                for j, note in enumerate(batch):
                    # ID unique base sur le chemin (pas le hash pour eviter les doublons)
                    path_hash = hashlib.md5(note['path'].encode()).hexdigest()[:12]
                    doc_id = f"note_{path_hash}"
                    ids.append(doc_id)
                    documents.append(note['content'])

                    # Metadonnees
                    meta = {
                        'path': note['path'],
                        'vault_path': note.get('vault_path', ''),
                        'title': note['title'],
                        'tags': ','.join(note['tags']),
                        'wikilinks': ','.join(note['wikilinks']),
                        'hash': note['hash'],
                        'modified': note['modified'],
                        'indexed_at': datetime.now().isoformat()
                    }
                    metadatas.append(meta)

                # Supprimer les anciennes versions
                for note in batch:
                    try:
                        result = self.collection.get(
                            where={"path": note['path']},
                            include=[]
                        )
                        if result['ids']:
                            self.collection.delete(ids=result['ids'])
                    except Exception:
                        pass

                # Ajouter les nouvelles versions
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )

                stats['indexed'] += len(batch)
                logger.info(f"Indexe {stats['indexed']}/{len(notes)} notes")

            except Exception as e:
                logger.error(f"Erreur batch {i}: {e}")
                stats['errors'] += len(batch)

    def get_stats(self) -> dict:
        """Retourne les statistiques de la base."""
        count = self.collection.count()
        graph_stats = self.graph.stats()

        return {
            'collection': self.collection_name,
            'vault_path': str(self.vault_path),
            'indexed_notes': count,
            'graph': graph_stats
        }

    def clear(self):
        """Vide completement la base."""
        self.chroma.delete_collection(self.collection_name)
        self.collection = self.chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.graph = WikilinkGraph()
        if self.graph_path.exists():
            self.graph_path.unlink()
        logger.info("Base videe")


def main():
    """CLI pour indexation."""
    import argparse

    parser = argparse.ArgumentParser(description="Indexeur Obsidian")
    parser.add_argument("vault", help="Chemin vers le vault")
    parser.add_argument("--db", default="./chroma_db", help="Chemin base de donnees")
    parser.add_argument("--full", action="store_true", help="Indexation complete (pas incremental)")
    parser.add_argument("--clear", action="store_true", help="Vider la base avant indexation")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    indexer = ObsidianIndexer(
        vault_path=args.vault,
        db_path=args.db
    )

    if args.clear:
        indexer.clear()

    stats = indexer.index_vault(incremental=not args.full)
    print(f"\nStatistiques:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
