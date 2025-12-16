"""Graphe de wikilinks pour gestion des backlinks."""

import json
from pathlib import Path
from typing import Optional
from collections import defaultdict


class WikilinkGraph:
    """
    Graphe bidirectionnel des liens entre notes Obsidian.

    Permet de:
    - Trouver les backlinks (notes qui pointent vers une note)
    - Trouver les liens sortants (notes vers lesquelles pointe une note)
    - Detecter les liens orphelins
    """

    def __init__(self):
        # note_path -> set of notes it links to
        self.outgoing: dict[str, set[str]] = defaultdict(set)

        # note_path -> set of notes that link to it
        self.incoming: dict[str, set[str]] = defaultdict(set)

        # Mapping titre/alias -> chemin reel
        self.title_to_path: dict[str, str] = {}

    def add_note(self, note_path: str, title: str, wikilinks: list[str]):
        """
        Ajoute une note au graphe.

        Args:
            note_path: Chemin relatif de la note (sans .md)
            title: Titre de la note
            wikilinks: Liste des [[liens]] dans la note
        """
        # Normaliser le chemin
        note_key = self._normalize_path(note_path)

        # Enregistrer le mapping titre -> chemin
        self.title_to_path[title.lower()] = note_key
        self.title_to_path[note_key.lower()] = note_key

        # Reset les liens sortants pour cette note
        old_links = self.outgoing.get(note_key, set())
        for old_target in old_links:
            self.incoming[old_target].discard(note_key)

        self.outgoing[note_key] = set()

        # Ajouter les nouveaux liens
        for link in wikilinks:
            target = self._resolve_link(link)
            self.outgoing[note_key].add(target)
            self.incoming[target].add(note_key)

    def remove_note(self, note_path: str):
        """Retire une note du graphe."""
        note_key = self._normalize_path(note_path)

        # Retirer des liens sortants
        for target in self.outgoing.get(note_key, []):
            self.incoming[target].discard(note_key)

        # Retirer des liens entrants
        for source in self.incoming.get(note_key, []):
            self.outgoing[source].discard(note_key)

        # Supprimer la note
        self.outgoing.pop(note_key, None)
        self.incoming.pop(note_key, None)

    def get_backlinks(self, note_path: str) -> list[str]:
        """
        Retourne les notes qui pointent vers cette note.

        Args:
            note_path: Chemin ou titre de la note

        Returns:
            Liste des chemins des notes qui pointent vers celle-ci
        """
        note_key = self._resolve_link(note_path)
        return sorted(list(self.incoming.get(note_key, set())))

    def get_outgoing_links(self, note_path: str) -> list[str]:
        """
        Retourne les notes vers lesquelles pointe cette note.

        Args:
            note_path: Chemin de la note

        Returns:
            Liste des chemins des notes liees
        """
        note_key = self._normalize_path(note_path)
        return sorted(list(self.outgoing.get(note_key, set())))

    def get_orphan_notes(self) -> list[str]:
        """Retourne les notes sans aucun lien entrant."""
        all_notes = set(self.outgoing.keys())
        linked_notes = set()
        for targets in self.outgoing.values():
            linked_notes.update(targets)

        orphans = []
        for note in all_notes:
            if not self.incoming.get(note):
                orphans.append(note)

        return sorted(orphans)

    def get_broken_links(self) -> list[tuple[str, str]]:
        """
        Retourne les liens vers des notes inexistantes.

        Returns:
            Liste de tuples (source_note, broken_link)
        """
        all_notes = set(self.outgoing.keys())
        broken = []

        for source, targets in self.outgoing.items():
            for target in targets:
                if target not in all_notes:
                    broken.append((source, target))

        return broken

    def _normalize_path(self, path: str) -> str:
        """Normalise un chemin de note."""
        # Retirer l'extension .md si presente
        if path.endswith('.md'):
            path = path[:-3]

        # Convertir les backslashes
        path = path.replace('\\', '/')

        return path

    def _resolve_link(self, link: str) -> str:
        """
        Resout un lien vers le chemin reel de la note.

        Gere:
        - Liens par titre: [[Mon Titre]]
        - Liens par chemin: [[folder/note]]
        """
        normalized = self._normalize_path(link)

        # Essayer de trouver par titre
        if normalized.lower() in self.title_to_path:
            return self.title_to_path[normalized.lower()]

        return normalized

    def rebuild_from_notes(self, notes: list[dict]):
        """
        Reconstruit le graphe depuis une liste de notes parsees.

        Args:
            notes: Liste de dicts avec 'vault_path', 'title', 'wikilinks'
        """
        self.outgoing.clear()
        self.incoming.clear()
        self.title_to_path.clear()

        for note in notes:
            vault_path = note.get('vault_path', note.get('path', ''))
            title = note.get('title', '')
            wikilinks = note.get('wikilinks', [])

            self.add_note(vault_path, title, wikilinks)

    def save(self, path: str):
        """Sauvegarde le graphe en JSON."""
        data = {
            'outgoing': {k: list(v) for k, v in self.outgoing.items()},
            'incoming': {k: list(v) for k, v in self.incoming.items()},
            'title_to_path': self.title_to_path
        }

        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

    def load(self, path: str) -> bool:
        """Charge le graphe depuis un fichier JSON."""
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))

            self.outgoing = defaultdict(set)
            for k, v in data.get('outgoing', {}).items():
                self.outgoing[k] = set(v)

            self.incoming = defaultdict(set)
            for k, v in data.get('incoming', {}).items():
                self.incoming[k] = set(v)

            self.title_to_path = data.get('title_to_path', {})

            return True
        except Exception:
            return False

    def stats(self) -> dict:
        """Retourne des statistiques sur le graphe."""
        total_notes = len(self.outgoing)
        total_links = sum(len(v) for v in self.outgoing.values())
        orphans = len(self.get_orphan_notes())
        broken = len(self.get_broken_links())

        return {
            'total_notes': total_notes,
            'total_links': total_links,
            'orphan_notes': orphans,
            'broken_links': broken,
            'avg_links_per_note': total_links / total_notes if total_notes > 0 else 0
        }


if __name__ == "__main__":
    # Test
    graph = WikilinkGraph()

    # Simuler quelques notes
    graph.add_note("index", "Index", ["note1", "note2"])
    graph.add_note("note1", "Note 1", ["note2", "note3"])
    graph.add_note("note2", "Note 2", ["index"])

    print("Backlinks vers note2:", graph.get_backlinks("note2"))
    print("Liens sortants de index:", graph.get_outgoing_links("index"))
    print("Stats:", graph.stats())
