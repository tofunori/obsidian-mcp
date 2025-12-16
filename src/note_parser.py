"""Parser pour notes Obsidian - extraction frontmatter, wikilinks, tags."""

import os
import re
import hashlib
from pathlib import Path
from typing import Optional
import yaml


def extract_frontmatter(content: str) -> tuple[dict, str]:
    """
    Extrait le frontmatter YAML d'une note.

    Returns:
        tuple: (frontmatter dict, contenu sans frontmatter)
    """
    if not content.startswith('---'):
        return {}, content

    # Chercher la fin du frontmatter
    end_match = re.search(r'\n---\s*\n', content[3:])
    if not end_match:
        return {}, content

    yaml_content = content[3:end_match.start() + 3]
    rest_content = content[end_match.end() + 3:]

    try:
        frontmatter = yaml.safe_load(yaml_content) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, rest_content


def extract_wikilinks(content: str) -> list[str]:
    """
    Extrait tous les [[wikilinks]] d'une note.

    Gere les formats:
    - [[note]]
    - [[note|alias]]
    - [[folder/note]]
    """
    pattern = r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]'
    matches = re.findall(pattern, content)

    # Normaliser les chemins (enlever .md si present)
    links = []
    for match in matches:
        link = match.strip()
        if link.endswith('.md'):
            link = link[:-3]
        links.append(link)

    return list(set(links))  # Deduplicate


def extract_tags(content: str, frontmatter: dict = None) -> list[str]:
    """
    Extrait tous les #tags d'une note.

    Combine:
    - Tags inline (#tag)
    - Tags du frontmatter (tags: [...])
    """
    tags = set()

    # Tags du frontmatter
    if frontmatter:
        fm_tags = frontmatter.get('tags', [])
        if isinstance(fm_tags, list):
            tags.update(fm_tags)
        elif isinstance(fm_tags, str):
            tags.add(fm_tags)

    # Tags inline (exclure les headers et les liens)
    # Pattern: #tag mais pas ##header ni #[[link]]
    pattern = r'(?<![#\w])#([a-zA-Z][a-zA-Z0-9_/-]*)'
    matches = re.findall(pattern, content)
    tags.update(matches)

    return sorted(list(tags))


def extract_title(content: str, frontmatter: dict = None, filename: str = None) -> str:
    """
    Extrait le titre d'une note.

    Priorite:
    1. Frontmatter 'title'
    2. Premier header H1
    3. Nom du fichier
    """
    # 1. Frontmatter
    if frontmatter and frontmatter.get('title'):
        return frontmatter['title']

    # 2. Premier H1
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    # 3. Nom du fichier
    if filename:
        return Path(filename).stem

    return "Sans titre"


def compute_hash(content: str) -> str:
    """Calcule le hash MD5 du contenu pour detection de changements."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def parse_note(path: str) -> Optional[dict]:
    """
    Parse une note Obsidian complete.

    Args:
        path: Chemin absolu vers le fichier .md

    Returns:
        dict avec: path, title, content, frontmatter, tags, wikilinks, modified, hash
    """
    path = Path(path)

    if not path.exists() or not path.suffix == '.md':
        return None

    try:
        content = path.read_text(encoding='utf-8')
    except Exception:
        return None

    frontmatter, body = extract_frontmatter(content)

    return {
        "path": str(path),
        "relative_path": path.name,
        "title": extract_title(body, frontmatter, path.name),
        "content": body,
        "raw_content": content,
        "frontmatter": frontmatter,
        "tags": extract_tags(body, frontmatter),
        "wikilinks": extract_wikilinks(body),
        "modified": os.path.getmtime(path),
        "hash": compute_hash(content),
        "size": len(content)
    }


def scan_vault(vault_path: str, exclude_folders: list[str] = None) -> list[dict]:
    """
    Scanne un vault Obsidian et parse toutes les notes.

    Args:
        vault_path: Chemin vers le vault
        exclude_folders: Dossiers a ignorer (ex: ['.obsidian', '.trash'])

    Returns:
        Liste de notes parsees
    """
    vault = Path(vault_path)
    exclude = set(exclude_folders or ['.obsidian', '.trash', '.git'])

    notes = []

    for md_file in vault.rglob('*.md'):
        # Verifier si dans un dossier exclu
        parts = md_file.relative_to(vault).parts
        if any(part in exclude for part in parts):
            continue

        note = parse_note(str(md_file))
        if note:
            # Ajouter le chemin relatif au vault
            note['vault_path'] = str(md_file.relative_to(vault))
            notes.append(note)

    return notes


if __name__ == "__main__":
    # Test rapide
    import sys

    if len(sys.argv) > 1:
        vault_path = sys.argv[1]
        notes = scan_vault(vault_path)
        print(f"Trouve {len(notes)} notes dans {vault_path}")

        for note in notes[:5]:
            print(f"\n--- {note['title']} ---")
            print(f"  Tags: {note['tags']}")
            print(f"  Links: {note['wikilinks']}")
