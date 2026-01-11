# obsidian-mcp

MCP server for Obsidian vaults with hybrid semantic search.

## Features

- Hybrid search: BM25 + Voyage AI embeddings + Cohere reranking
- Backlinks: Pre-computed wikilink graph
- Similar notes: Discover hidden connections
- Full CRUD: Create, read, update, delete, move notes
- **Live indexing**: Index notes directly from Claude Code without restart

## Installation

```bash
cd obsidian-mcp
uv sync
```

## Configuration

1. Copy `.env.example` to `.env` and add your API keys:
```bash
VOYAGE_API_KEY=your_key
COHERE_API_KEY=your_key  # optional
```

2. Set your vault path in `config/settings.yaml`:
```yaml
vault:
  path: "D:\\path\\to\\your\\vault"
```

## Usage

### Recommended Workflow (via MCP)

The recommended workflow is to use the MCP tools directly from Claude Code:

```
1. write   -> Create or modify a note (auto-indexes by default)
2. search  -> Note is instantly searchable
```

Notes are automatically indexed after `write`, so manual indexing is only needed for:
- Notes created manually in Obsidian (use `index` tool)
- Bulk imports (use `index` with `full=True`)

### Interactive Menu
```bash
python obsidian-menu.py
```

### CLI
```bash
# Full indexation
python obsidian-cli.py --vault "/path/to/vault" index --full

# Incremental indexation
python obsidian-cli.py --vault "/path/to/vault" index

# Status
python obsidian-cli.py --vault "/path/to/vault" status

# Search
python obsidian-cli.py --vault "/path/to/vault" search "query"
```

> **Note**: CLI indexation requires a Claude Code restart for changes to be visible in MCP searches due to ChromaDB caching. Use the `index` MCP tool instead for live updates.

## Claude Code Configuration

### Option 1: HTTP/SSE Server (Recommended)

Run as a shared HTTP server to avoid duplicating processes across Claude sessions.

**1. Start the HTTP server:**
```bash
uv run python -m src.server_http
# Server runs on http://127.0.0.1:8322/mcp
```

**2. Configure Claude Code** (`~/.claude.json`):
```json
{
  "mcpServers": {
    "obsidian": {
      "type": "sse",
      "url": "http://127.0.0.1:8322/mcp"
    }
  }
}
```

**3. (Optional) Run as systemd service** (Linux):
```bash
# Create ~/.config/systemd/user/obsidian-mcp.service
[Unit]
Description=Obsidian MCP HTTP Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/obsidian-mcp
Environment="VOYAGE_API_KEY=your_key"
Environment="COHERE_API_KEY=your_key"
Environment="OBSIDIAN_VAULT=/path/to/vault"
ExecStart=/path/to/uv run python -m src.server_http
Restart=on-failure

[Install]
WantedBy=default.target

# Enable and start
systemctl --user daemon-reload
systemctl --user enable --now obsidian-mcp
```

### Option 2: Stdio (per-session)

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "obsidian": {
      "command": "uv",
      "args": ["--directory", "/path/to/obsidian-mcp", "run", "python", "-m", "src.server"],
      "env": {
        "VOYAGE_API_KEY": "your_key",
        "COHERE_API_KEY": "your_key",
        "OBSIDIAN_VAULT": "/path/to/vault"
      }
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search` | Hybrid semantic search (BM25 + embeddings + reranking) |
| `read` | Read note with metadata and frontmatter |
| `write` | Create or update note (auto-indexes by default) |
| `delete` | Delete note |
| `move` | Move or rename note |
| `list` | List notes with folder/tag filters |
| `backlinks` | Find notes linking to a note |
| `similar` | Find semantically similar notes |
| `index` | Index new/modified notes (live, no restart) |
| `refresh` | Rebuild BM25 index from ChromaDB |
| `reload` | Reset all caches and connections |

### Auto-Indexing (Write Tool)

By default, `write` automatically indexes the note after creating or modifying it:

```python
# Auto-indexes after writing (default)
write(path="Notes/my_note.md", content="...")

# Disable auto-indexing if needed
write(path="Notes/my_note.md", content="...", auto_index=False)
```

This makes new notes instantly searchable without manual indexing.

### Index Tool

The `index` tool indexes notes directly within the MCP server process, avoiding ChromaDB cache synchronization issues:

```python
# Incremental indexation (default)
index()  # Only indexes new/modified notes

# Full reindexation
index(full=True)  # Reindexes all notes
```

**Why use `index` instead of CLI?**

ChromaDB maintains an HNSW index cache per process. When you index via CLI (separate process), the MCP server doesn't see the changes until restart. The `index` tool solves this by:
1. Running indexation in the same process as the MCP server
2. Resetting internal caches after indexation
3. Making new notes immediately searchable

## Project Structure

```
obsidian-mcp/
  src/
    server.py           # MCP server (11 tools)
    indexer.py          # Indexation engine
    retriever.py        # Hybrid search (BM25 + semantic + RRF)
    note_parser.py      # Obsidian markdown parser
    wikilink_graph.py   # Backlink graph
  config/
    settings.yaml       # Configuration
  chroma_db/            # Vector database (ChromaDB)
  obsidian-cli.py       # CLI interface
  obsidian-menu.py      # Interactive menu
```

## Technical Details

### Hybrid Search Pipeline

1. **BM25**: Lexical search for exact term matching
2. **Semantic**: Voyage AI embeddings (`voyage-3`) for meaning
3. **Fusion**: Reciprocal Rank Fusion (RRF) combines results
4. **Reranking**: Cohere reranker (`rerank-v4.0-pro`) for final ordering

### Embedding Model

- Model: `voyage-3`
- Dimensions: 1024
- Provider: Voyage AI

### Database

- ChromaDB with persistent storage
- SQLite backend with WAL mode
- HNSW index for fast similarity search

## License

MIT
