import os
from mdutils.mdutils import MdUtils
from rich.console import Console
from rich.progress import track

console = Console()

# Verzeichnisse, die ausgeschlossen werden
EXCLUDED_DIRS = {"__pycache__", ".git", ".venv", "build", "dist"}

def collect_python_files(root_dir):
    python_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Verzeichnisse filtern
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for filename in filenames:
            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                python_files.append(full_path)

    return python_files

def write_markdown(python_files, output_file):
    md_file = MdUtils(file_name=output_file.replace(".md", ""))
    md_file.new_header(level=1, title='Python Files Collection')

    for filepath in track(sorted(python_files), description="📄 Verarbeite Dateien..."):
        md_file.new_header(level=2, title=f'{filepath}')

        try:
            with open(filepath, "r", encoding="utf-8") as py_file:
                content = py_file.read()

            # Code Block erzeugen
            md_file.new_line("```python")
            md_file.new_line(content)
            md_file.new_line("```")
            md_file.new_line()
        except Exception as e:
            md_file.new_paragraph(f"**Fehler beim Lesen der Datei:** {e}")

    # Speichern
    md_file.create_md_file()

def main():
    root_dir = "./"
    output_file = "python_files_collection.md"

    console.print(f"[bold cyan]Scanne Verzeichnis:[/] {root_dir}")
    python_files = collect_python_files(root_dir)
    console.print(f"[bold green]{len(python_files)} Python-Dateien gefunden.[/]")

    console.print(f"[bold cyan]Schreibe Markdown-Datei:[/] {output_file}")
    write_markdown(python_files, output_file)
    console.print("[bold green]Fertig![/]")

if __name__ == "__main__":
    main()
