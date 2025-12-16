# obsidian-mcp

MCP server for Obsidian vaults with hybrid semantic search.

## Features

- Hybrid search: BM25 + Voyage AI embeddings + Cohere reranking
- Backlinks: Pre-computed wikilink graph
- Similar notes: Discover hidden connections
- Full CRUD: Create, read, update, delete, move notes

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

## Claude Code Configuration

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
| `search` | Hybrid semantic search |
| `read` | Read note with metadata |
| `write` | Create or update note |
| `delete` | Delete note |
| `move` | Move or rename note |
| `list` | List notes with filters |
| `backlinks` | Find notes linking to a note |
| `similar` | Find semantically similar notes |

## Project Structure

```
obsidian-mcp/
  src/
    server.py           # MCP server (8 tools)
    note_parser.py      # Obsidian parser
    wikilink_graph.py   # Link graph
    retriever.py        # Hybrid search
    indexer.py          # Indexation
  config/
    settings.yaml       # Configuration
  chroma_db/            # Vector database
  obsidian-cli.py       # CLI
  obsidian-menu.py      # Interactive menu
```

## License

MIT
