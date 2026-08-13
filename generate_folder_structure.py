from pathlib import Path


def generate_folder_structure(
    target_folder,
    output_file="folder_structure.md",
    omit_folders=None,
):
    """
    Generate a Markdown representation of a folder structure.

    Parameters
    ----------
    target_folder : str or Path
        Root folder to scan.

    output_file : str
        Markdown file to write.

    omit_folders : list[str], optional
        List of folders (relative to target_folder) to completely skip.

        Example:
        [
            "venv",
            "__pycache__",
            "data/raw",
            "logs/archive"
        ]
    """

    root = Path(target_folder).resolve()

    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")

    omit_folders = omit_folders or []

    # Convert omitted folders to absolute paths
    omitted = {(root / Path(folder)).resolve() for folder in omit_folders}

    lines = [f"# Folder Structure\n", f"Root: `{root.name}`\n", "```text"]

    def walk(directory, prefix=""):
        entries = sorted(
            directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower())
        )

        # Remove omitted folders
        entries = [
            e
            for e in entries
            if not any(
                e.resolve() == omit or omit in e.resolve().parents for omit in omitted
            )
        ]

        for index, entry in enumerate(entries):
            connector = "└── " if index == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")

            if entry.is_dir():
                extension = "    " if index == len(entries) - 1 else "│   "
                walk(entry, prefix + extension)

    lines.append(root.name)
    walk(root)
    lines.append("```")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown written to: {output_file}")


generate_folder_structure(
    target_folder=r"C:\PY_Programs\MSc_Project",
    output_file="agent_folder_structure.md",
    omit_folders=[
        ".venv",
        ".git",
        "__pycache__",
        "POC/__pycache__",
        "POC/logs",
        "POC/test_module/__pycache__",
        "agent_framework/archive",
        "backend_server/logs",
        "backend_server/node_modules",
    ],
)
