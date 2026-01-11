#!/usr/bin/env python3
"""
Serveur HTTP/SSE pour obsidian-mcp (partage entre sessions).
Port: 8322

Utilise fastmcp standalone (pas mcp.server.fastmcp) pour supporter host/port.
"""

import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use standalone fastmcp (supports host/port in run())
from fastmcp import FastMCP

# Import all the tool implementations from server
from src import server

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Create new FastMCP server with standalone package
mcp = FastMCP("obsidian-mcp")


# Re-register all tools from the original server
@mcp.tool()
def search(query: str, top_k: int = 10, folder: str = None, tags: str = None) -> str:
    """Recherche semantique hybride dans le vault Obsidian."""
    return server.search(query, top_k, folder, tags)


@mcp.tool()
def read(path: str) -> str:
    """Lit une note avec ses metadonnees."""
    return server.read(path)


@mcp.tool()
def write(path: str, content: str, mode: str = "replace", auto_index: bool = True) -> str:
    """Cree ou modifie une note."""
    return server.write(path, content, mode, auto_index)


@mcp.tool()
def delete(path: str) -> str:
    """Supprime une note."""
    return server.delete(path)


@mcp.tool()
def move(old_path: str, new_path: str) -> str:
    """Deplace ou renomme une note."""
    return server.move(old_path, new_path)


@mcp.tool()
def list(folder: str = None, tags: str = None, limit: int = 50) -> str:
    """Liste les notes du vault."""
    return server.list(folder, tags, limit)


@mcp.tool()
def backlinks(path: str) -> str:
    """Trouve les notes qui pointent vers cette note."""
    return server.backlinks(path)


@mcp.tool()
def similar(path: str, top_k: int = 5) -> str:
    """Trouve les notes semantiquement similaires."""
    return server.similar(path, top_k)


@mcp.tool()
def refresh() -> str:
    """Rafraichit l'index BM25 apres indexation."""
    return server.refresh()


@mcp.tool()
def index(full: bool = False) -> str:
    """Indexe les nouvelles notes."""
    return server.index(full)


@mcp.tool()
def clear() -> str:
    """Vide completement la base ChromaDB et le graphe."""
    return server.clear()


@mcp.tool()
def reload() -> str:
    """Recharge completement tous les caches et connexions."""
    return server.reload()


def main():
    """Point d'entree du serveur HTTP/SSE."""
    print("=" * 60)
    print("OBSIDIAN MCP HTTP SERVER")
    print("=" * 60)
    print("Starting on http://127.0.0.1:8322/sse")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    logger.info("Demarrage obsidian-mcp en mode HTTP/SSE...")
    mcp.run(transport="sse", host="127.0.0.1", port=8322)


if __name__ == "__main__":
    main()
