"""
Serveur MCP pour vault Obsidian.
8 outils: search, read, write, delete, move, list, backlinks, similar
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import yaml
import chromadb
from chromadb.config import Settings
import voyageai
import cohere
from mcp.server.fastmcp import FastMCP

# Configure logging to stderr (keep stdout clean for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("obsidian-mcp")

# Global clients (lazy initialized)
_voyage_client = None
_cohere_client = None
_chroma_client = None
_collection = None
_retriever = None
_graph = None
_config = None


def get_config() -> dict:
    """Charge la configuration."""
    global _config
    if _config is not None:
        return _config

    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
    else:
        _config = {
            'vault': {'path': os.getenv('OBSIDIAN_VAULT', '')},
            'embedding': {'model': 'voyage-context-3'},
            'reranking': {'model': 'rerank-v4.0-pro', 'top_n': 10},
            'database': {'path': './chroma_db', 'collection': 'obsidian_notes'},
            'search': {'default_top_k': 10, 'bm25_weight': 0.3}
        }

    # Override with environment variables
    if os.getenv('OBSIDIAN_VAULT'):
        _config['vault']['path'] = os.getenv('OBSIDIAN_VAULT')

    return _config


def get_vault_path() -> Path:
    """Retourne le chemin du vault."""
    config = get_config()
    return Path(config['vault']['path'])


def get_voyage_client():
    """Lazy init Voyage AI client."""
    global _voyage_client
    if _voyage_client is None:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY non defini")
        _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


def get_cohere_client():
    """Lazy init Cohere client."""
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            _cohere_client = cohere.Client(api_key=api_key)
    return _cohere_client


def get_collection():
    """Lazy init ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is None:
        config = get_config()
        db_path = Path(__file__).parent.parent / config['database']['path']

        _chroma_client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _chroma_client.get_or_create_collection(
            name=config['database']['collection'],
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def get_retriever():
    """Lazy init retriever."""
    global _retriever
    if _retriever is None:
        from .retriever import ObsidianRetriever

        collection = get_collection()
        voyage = get_voyage_client()

        def embed_fn(texts):
            # Utilise voyage-3 pour les requetes (meme modele que l'indexation)
            result = voyage.embed(texts=texts, model="voyage-3", input_type="query")
            return result.embeddings

        _retriever = ObsidianRetriever(collection, embedding_function=embed_fn)
    return _retriever


def get_graph():
    """Lazy init wikilink graph."""
    global _graph
    if _graph is None:
        from .wikilink_graph import WikilinkGraph

        config = get_config()
        db_path = Path(__file__).parent.parent / config['database']['path']
        graph_path = db_path / "wikilink_graph.json"

        _graph = WikilinkGraph()
        if graph_path.exists():
            _graph.load(str(graph_path))
    return _graph


def rerank_results(query: str, results: list[dict], top_n: int = 10) -> list[dict]:
    """Rerank avec Cohere si disponible."""
    cohere_client = get_cohere_client()
    if not cohere_client or not results:
        return results[:top_n]

    try:
        config = get_config()
        model = config.get('reranking', {}).get('model', 'rerank-v3.5')

        documents = [r['text'][:4000] for r in results]  # Limite de tokens

        response = cohere_client.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=min(top_n, len(documents))
        )

        reranked = []
        for item in response.results:
            result = results[item.index].copy()
            result['rerank_score'] = item.relevance_score
            reranked.append(result)

        return reranked

    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return results[:top_n]


# ============================================================================
# MCP TOOLS (8 outils)
# ============================================================================

@mcp.tool()
def search(
    query: str,
    top_k: int = 10,
    folder: Optional[str] = None,
    tags: Optional[str] = None
) -> str:
    """
    Recherche semantique hybride dans le vault Obsidian.

    Args:
        query: Requete de recherche
        top_k: Nombre de resultats (defaut: 10)
        folder: Filtrer par dossier (optionnel)
        tags: Filtrer par tags, separes par virgules (optionnel)

    Returns:
        Notes pertinentes avec titre, extrait et score
    """
    retriever = get_retriever()
    config = get_config()

    # Parser les tags
    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(',')]

    # Recherche hybride
    alpha = 1.0 - config.get('search', {}).get('bm25_weight', 0.3)
    results = retriever.search(
        query=query,
        top_k=top_k * 2,  # Over-fetch for reranking
        alpha=alpha,
        folder=folder,
        tags=tag_list
    )

    # Rerank
    results = rerank_results(query, results, top_n=top_k)

    # Formater la sortie
    output = []
    for i, r in enumerate(results, 1):
        meta = r.get('metadata', {})
        title = meta.get('title', 'Sans titre')
        path = meta.get('vault_path', meta.get('path', ''))
        note_tags = meta.get('tags', '')
        excerpt = r['text'][:300] + "..." if len(r['text']) > 300 else r['text']
        score = r.get('rerank_score', r.get('score', 0))

        output.append(f"## {i}. {title}")
        output.append(f"**Chemin**: {path}")
        if note_tags:
            output.append(f"**Tags**: {note_tags}")
        output.append(f"**Score**: {score:.3f}")
        output.append(f"\n{excerpt}\n")

    if not output:
        return "Aucun resultat trouve."

    return "\n".join(output)


@mcp.tool()
def read(path: str) -> str:
    """
    Lit une note avec ses metadonnees.

    Args:
        path: Chemin de la note (relatif au vault ou absolu)

    Returns:
        Contenu de la note avec frontmatter, tags et liens
    """
    vault = get_vault_path()

    # Resoudre le chemin
    note_path = Path(path)
    if not note_path.is_absolute():
        note_path = vault / path

    if not note_path.suffix:
        note_path = note_path.with_suffix('.md')

    if not note_path.exists():
        return f"Note non trouvee: {path}"

    try:
        from .note_parser import parse_note

        note = parse_note(str(note_path))
        if not note:
            return f"Impossible de parser: {path}"

        output = []
        output.append(f"# {note['title']}")
        output.append(f"**Chemin**: {note_path.relative_to(vault)}")

        if note['tags']:
            output.append(f"**Tags**: {', '.join(note['tags'])}")

        if note['wikilinks']:
            output.append(f"**Liens**: {', '.join(note['wikilinks'][:10])}")

        if note['frontmatter']:
            output.append("\n**Frontmatter**:")
            for key, value in note['frontmatter'].items():
                output.append(f"  - {key}: {value}")

        output.append("\n---\n")
        output.append(note['content'])

        return "\n".join(output)

    except Exception as e:
        return f"Erreur lecture: {e}"


@mcp.tool()
def write(
    path: str,
    content: str,
    mode: str = "replace"
) -> str:
    """
    Cree ou modifie une note.

    Args:
        path: Chemin de la note (relatif au vault)
        content: Contenu a ecrire
        mode: "create" (nouvelle note), "replace" (remplacer), "append" (ajouter)

    Returns:
        Confirmation ou erreur
    """
    vault = get_vault_path()
    note_path = vault / path

    if not note_path.suffix:
        note_path = note_path.with_suffix('.md')

    try:
        # Creer les dossiers parents si necessaire
        note_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "create":
            if note_path.exists():
                return f"La note existe deja: {path}. Utilisez mode='replace' pour remplacer."
            note_path.write_text(content, encoding='utf-8')
            return f"Note creee: {path}"

        elif mode == "append":
            existing = ""
            if note_path.exists():
                existing = note_path.read_text(encoding='utf-8')
            note_path.write_text(existing + "\n" + content, encoding='utf-8')
            return f"Contenu ajoute a: {path}"

        else:  # replace
            note_path.write_text(content, encoding='utf-8')
            return f"Note mise a jour: {path}"

    except Exception as e:
        return f"Erreur ecriture: {e}"


@mcp.tool()
def delete(path: str) -> str:
    """
    Supprime une note.

    Args:
        path: Chemin de la note (relatif au vault)

    Returns:
        Confirmation ou erreur
    """
    vault = get_vault_path()
    note_path = vault / path

    if not note_path.suffix:
        note_path = note_path.with_suffix('.md')

    if not note_path.exists():
        return f"Note non trouvee: {path}"

    try:
        note_path.unlink()
        return f"Note supprimee: {path}"
    except Exception as e:
        return f"Erreur suppression: {e}"


@mcp.tool()
def move(old_path: str, new_path: str) -> str:
    """
    Deplace ou renomme une note.

    Args:
        old_path: Chemin actuel de la note
        new_path: Nouveau chemin

    Returns:
        Confirmation ou erreur
    """
    vault = get_vault_path()

    old_note = vault / old_path
    new_note = vault / new_path

    if not old_note.suffix:
        old_note = old_note.with_suffix('.md')
    if not new_note.suffix:
        new_note = new_note.with_suffix('.md')

    if not old_note.exists():
        return f"Note non trouvee: {old_path}"

    try:
        new_note.parent.mkdir(parents=True, exist_ok=True)
        old_note.rename(new_note)
        return f"Note deplacee: {old_path} -> {new_path}"
    except Exception as e:
        return f"Erreur deplacement: {e}"


@mcp.tool()
def list(
    folder: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 50
) -> str:
    """
    Liste les notes du vault.

    Args:
        folder: Filtrer par dossier (optionnel)
        tags: Filtrer par tags, separes par virgules (optionnel)
        limit: Nombre max de resultats (defaut: 50)

    Returns:
        Liste des notes avec titre et tags
    """
    from .note_parser import scan_vault

    vault = get_vault_path()
    scan_path = vault / folder if folder else vault

    if not scan_path.exists():
        return f"Dossier non trouve: {folder}"

    notes = scan_vault(str(scan_path))

    # Filtrer par tags
    if tags:
        tag_list = [t.strip().lower() for t in tags.split(',')]
        notes = [
            n for n in notes
            if any(t.lower() in [x.lower() for x in n['tags']] for t in tag_list)
        ]

    # Limiter
    notes = notes[:limit]

    if not notes:
        return "Aucune note trouvee."

    output = [f"**{len(notes)} notes trouvees:**\n"]
    for note in notes:
        tags_str = f" [{', '.join(note['tags'][:3])}]" if note['tags'] else ""
        output.append(f"- **{note['title']}**{tags_str}")
        output.append(f"  `{note.get('vault_path', note['path'])}`")

    return "\n".join(output)


@mcp.tool()
def backlinks(path: str) -> str:
    """
    Trouve les notes qui pointent vers cette note.

    Args:
        path: Chemin ou titre de la note

    Returns:
        Liste des notes avec backlinks
    """
    graph = get_graph()

    # Normaliser le chemin
    if path.endswith('.md'):
        path = path[:-3]

    links = graph.get_backlinks(path)

    if not links:
        return f"Aucun backlink trouve pour: {path}"

    output = [f"**{len(links)} notes pointent vers {path}:**\n"]
    for link in links:
        output.append(f"- [[{link}]]")

    return "\n".join(output)


@mcp.tool()
def similar(path: str, top_k: int = 5) -> str:
    """
    Trouve les notes semantiquement similaires.

    Args:
        path: Chemin de la note source
        top_k: Nombre de resultats (defaut: 5)

    Returns:
        Notes similaires avec score
    """
    retriever = get_retriever()

    # Normaliser le chemin
    vault = get_vault_path()
    full_path = vault / path
    if not full_path.suffix:
        full_path = full_path.with_suffix('.md')

    results = retriever.find_similar(str(full_path), top_k=top_k)

    if not results:
        return f"Aucune note similaire trouvee pour: {path}"

    output = [f"**Notes similaires a {path}:**\n"]
    for i, r in enumerate(results, 1):
        meta = r.get('metadata', {})
        title = meta.get('title', 'Sans titre')
        note_path = meta.get('vault_path', '')
        score = r.get('score', 0)

        output.append(f"{i}. **{title}** (score: {score:.3f})")
        output.append(f"   `{note_path}`")

    return "\n".join(output)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Point d'entree du serveur MCP."""
    logger.info("Demarrage obsidian-mcp...")
    mcp.run()


if __name__ == "__main__":
    main()
