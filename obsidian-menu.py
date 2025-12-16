#!/usr/bin/env python3
"""Menu interactif Rich pour obsidian-mcp."""

import os
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Configuration par defaut
VAULT_PATH = os.getenv("OBSIDIAN_VAULT", "")
DB_PATH = "./chroma_db"


def show_header():
    """Affiche l'en-tete."""
    console.print(Panel.fit(
        "[bold cyan]obsidian-mcp[/bold cyan]\n"
        "[dim]Serveur MCP pour vault Obsidian avec recherche semantique[/dim]",
        border_style="cyan"
    ))
    console.print()


def show_menu():
    """Affiche le menu principal."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="cyan bold")
    table.add_column("Description")

    table.add_row("1", "Indexer le vault")
    table.add_row("2", "Indexation incrementale")
    table.add_row("3", "Statut de l'indexation")
    table.add_row("4", "Reconstruire les liens")
    table.add_row("5", "Recherche de test")
    table.add_row("6", "Lister les notes")
    table.add_row("", "")
    table.add_row("c", "Configurer le vault")
    table.add_row("q", "Quitter")

    console.print(Panel(table, title="Menu principal", border_style="blue"))


def menu_index(full: bool = False):
    """Indexation du vault."""
    global VAULT_PATH, DB_PATH

    if not VAULT_PATH:
        console.print("[red]Vault non configure. Utilisez l'option 'c' pour configurer.[/red]")
        return

    from src.indexer import ObsidianIndexer

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Initialisation...", total=None)

        try:
            indexer = ObsidianIndexer(vault_path=VAULT_PATH, db_path=DB_PATH)

            progress.update(task, description="Indexation en cours...")
            stats = indexer.index_vault(incremental=not full)

            progress.update(task, description="Termine!")

        except Exception as e:
            console.print(f"[red]Erreur: {e}[/red]")
            return

    # Afficher les stats
    table = Table(title="Resultats de l'indexation")
    table.add_column("Metrique", style="cyan")
    table.add_column("Valeur", style="green")

    table.add_row("Notes totales", str(stats['total_notes']))
    table.add_row("Indexees", str(stats['indexed']))
    table.add_row("Ignorees", str(stats['skipped']))
    table.add_row("Supprimees", str(stats['deleted']))
    table.add_row("Erreurs", str(stats['errors']))

    console.print(table)


def menu_status():
    """Affiche le statut."""
    global VAULT_PATH, DB_PATH

    if not VAULT_PATH:
        console.print("[red]Vault non configure.[/red]")
        return

    from src.indexer import ObsidianIndexer

    try:
        indexer = ObsidianIndexer(vault_path=VAULT_PATH, db_path=DB_PATH)
        stats = indexer.get_stats()

        table = Table(title="Statut obsidian-mcp")
        table.add_column("Propriete", style="cyan")
        table.add_column("Valeur", style="green")

        table.add_row("Collection", stats['collection'])
        table.add_row("Vault", stats['vault_path'])
        table.add_row("Notes indexees", str(stats['indexed_notes']))

        console.print(table)

        if stats['graph']:
            console.print("\n[bold]Graphe de liens:[/bold]")
            for key, value in stats['graph'].items():
                console.print(f"  {key}: {value}")

    except Exception as e:
        console.print(f"[red]Erreur: {e}[/red]")


def menu_rebuild_links():
    """Reconstruit le graphe de liens."""
    global VAULT_PATH, DB_PATH

    if not VAULT_PATH:
        console.print("[red]Vault non configure.[/red]")
        return

    from src.note_parser import scan_vault
    from src.wikilink_graph import WikilinkGraph

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Scan du vault...", total=None)

        notes = scan_vault(VAULT_PATH)
        progress.update(task, description=f"Reconstruction ({len(notes)} notes)...")

        graph = WikilinkGraph()
        graph.rebuild_from_notes(notes)

        db_path = Path(DB_PATH)
        db_path.mkdir(parents=True, exist_ok=True)
        graph.save(str(db_path / "wikilink_graph.json"))

        progress.update(task, description="Termine!")

    stats = graph.stats()
    console.print(f"\n[green]Graphe reconstruit![/green]")
    for key, value in stats.items():
        console.print(f"  {key}: {value}")


def menu_search():
    """Recherche interactive."""
    global VAULT_PATH, DB_PATH

    if not VAULT_PATH:
        console.print("[red]Vault non configure.[/red]")
        return

    import chromadb
    from chromadb.config import Settings
    import voyageai
    from src.retriever import ObsidianRetriever

    query = Prompt.ask("Requete de recherche")
    if not query:
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Recherche...", total=None)

        try:
            chroma = chromadb.PersistentClient(
                path=str(DB_PATH),
                settings=Settings(anonymized_telemetry=False)
            )
            collection = chroma.get_or_create_collection("obsidian_notes")

            voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

            def embed_fn(texts):
                result = voyage.embed(texts=texts, model="voyage-3", input_type="query")
                return result.embeddings

            retriever = ObsidianRetriever(collection, embedding_function=embed_fn)
            results = retriever.search(query, top_k=10)

            progress.update(task, description="Termine!")

        except Exception as e:
            console.print(f"[red]Erreur: {e}[/red]")
            return

    if not results:
        console.print("[yellow]Aucun resultat trouve.[/yellow]")
        return

    table = Table(title=f"Resultats pour: {query}")
    table.add_column("#", style="dim")
    table.add_column("Titre", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Chemin", style="dim")

    for i, r in enumerate(results, 1):
        meta = r.get('metadata', {})
        table.add_row(
            str(i),
            meta.get('title', 'Sans titre'),
            f"{r['score']:.3f}",
            meta.get('vault_path', '')[:40]
        )

    console.print(table)


def menu_list():
    """Liste les notes."""
    global VAULT_PATH

    if not VAULT_PATH:
        console.print("[red]Vault non configure.[/red]")
        return

    from src.note_parser import scan_vault

    notes = scan_vault(VAULT_PATH)

    table = Table(title=f"{len(notes)} notes dans le vault")
    table.add_column("Titre", style="cyan")
    table.add_column("Tags", style="yellow")
    table.add_column("Liens", style="blue")

    for note in notes[:30]:
        table.add_row(
            note['title'][:40],
            ', '.join(note['tags'][:3]),
            str(len(note['wikilinks']))
        )

    console.print(table)

    if len(notes) > 30:
        console.print(f"[dim]... et {len(notes) - 30} autres notes[/dim]")


def menu_configure():
    """Configure le vault."""
    global VAULT_PATH, DB_PATH

    console.print(f"\n[bold]Configuration actuelle:[/bold]")
    console.print(f"  Vault: {VAULT_PATH or '[non configure]'}")
    console.print(f"  Base: {DB_PATH}")

    new_vault = Prompt.ask("Chemin du vault", default=VAULT_PATH)
    if new_vault and Path(new_vault).exists():
        VAULT_PATH = new_vault
        console.print(f"[green]Vault configure: {VAULT_PATH}[/green]")
    elif new_vault:
        console.print(f"[red]Chemin invalide: {new_vault}[/red]")


def main():
    """Boucle principale du menu."""
    global VAULT_PATH

    show_header()

    # Charger config depuis settings.yaml si disponible
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not VAULT_PATH and config.get('vault', {}).get('path'):
                VAULT_PATH = config['vault']['path']

    console.print(f"[dim]Vault: {VAULT_PATH or 'Non configure'}[/dim]\n")

    while True:
        show_menu()
        choice = Prompt.ask("Choix", choices=["1", "2", "3", "4", "5", "6", "c", "q"])

        console.print()

        if choice == "1":
            menu_index(full=True)
        elif choice == "2":
            menu_index(full=False)
        elif choice == "3":
            menu_status()
        elif choice == "4":
            menu_rebuild_links()
        elif choice == "5":
            menu_search()
        elif choice == "6":
            menu_list()
        elif choice == "c":
            menu_configure()
        elif choice == "q":
            console.print("[cyan]Au revoir![/cyan]")
            break

        console.print()
        Prompt.ask("Appuyez sur Entree pour continuer", default="")
        console.clear()
        show_header()


if __name__ == "__main__":
    main()
