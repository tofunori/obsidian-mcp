#!/usr/bin/env python3
"""CLI pour obsidian-mcp - Indexation et gestion du vault."""

import argparse
import os
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def cmd_index(args):
    """Indexe le vault Obsidian."""
    from src.indexer import ObsidianIndexer

    indexer = ObsidianIndexer(
        vault_path=args.vault,
        db_path=args.db
    )

    if args.clear:
        print("Vidage de la base...")
        indexer.clear()

    print(f"Indexation du vault: {args.vault}")
    stats = indexer.index_vault(incremental=not args.full)

    print("\nStatistiques:")
    print(f"  Notes totales: {stats['total_notes']}")
    print(f"  Indexees: {stats['indexed']}")
    print(f"  Ignorees (inchangees): {stats['skipped']}")
    print(f"  Supprimees: {stats['deleted']}")
    print(f"  Erreurs: {stats['errors']}")


def cmd_status(args):
    """Affiche le statut de l'indexation."""
    from src.indexer import ObsidianIndexer

    indexer = ObsidianIndexer(
        vault_path=args.vault,
        db_path=args.db
    )

    stats = indexer.get_stats()

    print("\nStatut obsidian-mcp:")
    print(f"  Collection: {stats['collection']}")
    print(f"  Vault: {stats['vault_path']}")
    print(f"  Notes indexees: {stats['indexed_notes']}")
    print(f"\nGraphe de liens:")
    for key, value in stats['graph'].items():
        print(f"  {key}: {value}")


def cmd_list(args):
    """Liste les notes indexees."""
    from src.note_parser import scan_vault

    notes = scan_vault(args.vault)

    if args.tags:
        tag_filter = [t.strip().lower() for t in args.tags.split(',')]
        notes = [
            n for n in notes
            if any(t.lower() in [x.lower() for x in n['tags']] for t in tag_filter)
        ]

    print(f"\n{len(notes)} notes trouvees:\n")

    for note in notes[:args.limit]:
        tags = f" [{', '.join(note['tags'][:3])}]" if note['tags'] else ""
        print(f"  - {note['title']}{tags}")
        print(f"    {note.get('vault_path', note['path'])}")


def cmd_search(args):
    """Recherche dans le vault."""
    import chromadb
    from chromadb.config import Settings
    import voyageai

    from src.retriever import ObsidianRetriever

    # Init
    db_path = Path(args.db)
    chroma = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma.get_or_create_collection("obsidian_notes")

    voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    def embed_fn(texts):
        result = voyage.embed(texts=texts, model="voyage-3", input_type="query")
        return result.embeddings

    retriever = ObsidianRetriever(collection, embedding_function=embed_fn)

    # Recherche
    print(f"\nRecherche: {args.query}\n")
    results = retriever.search(args.query, top_k=args.limit)

    for i, r in enumerate(results, 1):
        meta = r.get('metadata', {})
        print(f"{i}. {meta.get('title', 'Sans titre')} (score: {r['score']:.3f})")
        print(f"   {meta.get('vault_path', '')}")
        print(f"   {r['text'][:150]}...\n")


def cmd_rebuild_links(args):
    """Reconstruit le graphe de liens."""
    from src.note_parser import scan_vault
    from src.wikilink_graph import WikilinkGraph

    print(f"Scan du vault: {args.vault}")
    notes = scan_vault(args.vault)

    print(f"Reconstruction du graphe ({len(notes)} notes)...")
    graph = WikilinkGraph()
    graph.rebuild_from_notes(notes)

    db_path = Path(args.db)
    db_path.mkdir(parents=True, exist_ok=True)
    graph.save(str(db_path / "wikilink_graph.json"))

    stats = graph.stats()
    print(f"\nGraphe reconstruit:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI obsidian-mcp - Indexation et gestion du vault"
    )

    # Arguments globaux
    parser.add_argument(
        "--vault", "-v",
        default=os.getenv("OBSIDIAN_VAULT", ""),
        help="Chemin vers le vault Obsidian"
    )
    parser.add_argument(
        "--db", "-d",
        default="./chroma_db",
        help="Chemin vers la base de donnees"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commandes")

    # index
    p_index = subparsers.add_parser("index", help="Indexer le vault")
    p_index.add_argument("--full", action="store_true", help="Indexation complete")
    p_index.add_argument("--clear", action="store_true", help="Vider avant indexation")
    p_index.set_defaults(func=cmd_index)

    # status
    p_status = subparsers.add_parser("status", help="Statut de l'indexation")
    p_status.set_defaults(func=cmd_status)

    # list
    p_list = subparsers.add_parser("list", help="Lister les notes")
    p_list.add_argument("--tags", "-t", help="Filtrer par tags (comma-separated)")
    p_list.add_argument("--limit", "-l", type=int, default=50, help="Limite")
    p_list.set_defaults(func=cmd_list)

    # search
    p_search = subparsers.add_parser("search", help="Rechercher dans le vault")
    p_search.add_argument("query", help="Requete de recherche")
    p_search.add_argument("--limit", "-l", type=int, default=10, help="Limite")
    p_search.set_defaults(func=cmd_search)

    # rebuild-links
    p_links = subparsers.add_parser("rebuild-links", help="Reconstruire graphe de liens")
    p_links.set_defaults(func=cmd_rebuild_links)

    args = parser.parse_args()

    if not args.vault:
        print("Erreur: Vault non specifie. Utilisez --vault ou OBSIDIAN_VAULT")
        sys.exit(1)

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
