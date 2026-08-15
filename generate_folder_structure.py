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

    output_file : str or Path
        Markdown file to write.

    omit_folders : list[str], optional
        Folder names or relative folder paths to completely exclude.

        Matching is CASE-INSENSITIVE.

        Folder names are excluded ANYWHERE in the directory tree.

        Example:
        [
            ".venv",
            ".git",
            "__pycache__",
            "logs",
            "node_modules",
            "archive",
        ]

        This means:
            "logs" excludes:
                logs/
                Logs/
                LOGS/
                LoGs/

        You can also specify a relative path:

            "agent_framework/archive"

        This will exclude that specific path regardless of case, e.g.:

            agent_framework/archive
            Agent_Framework/Archive
            AGENT_FRAMEWORK/ARCHIVE
    """

    root = Path(target_folder).resolve()

    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")

    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory.")

    omit_folders = omit_folders or []

    # ---------------------------------------------------------
    # Separate global folder names from relative folder paths
    # ---------------------------------------------------------

    omitted_names = set()
    omitted_relative_paths = set()

    for folder in omit_folders:
        folder_path = Path(folder)

        # -----------------------------------------------------
        # Folder name only:
        #
        # "__pycache__"
        # "logs"
        # "node_modules"
        #
        # Excluded anywhere in the project.
        # -----------------------------------------------------

        if len(folder_path.parts) == 1:
            omitted_names.add(folder_path.name.casefold())

        # -----------------------------------------------------
        # Relative path:
        #
        # "agent_framework/archive"
        # "POC/logs"
        #
        # Excludes only that relative path.
        # -----------------------------------------------------

        else:
            normalized_relative_path = "/".join(
                part.casefold() for part in folder_path.parts
            )

            omitted_relative_paths.add(normalized_relative_path)

    # ---------------------------------------------------------
    # Convert a path to a case-insensitive relative path
    # ---------------------------------------------------------

    def get_normalized_relative_path(path):
        """
        Convert a path into a normalized, case-insensitive path
        relative to the root folder.
        """

        relative_path = path.resolve().relative_to(root)

        return "/".join(part.casefold() for part in relative_path.parts)

    # ---------------------------------------------------------
    # Check whether a directory should be excluded
    # ---------------------------------------------------------

    def should_omit(path):
        """
        Return True if the given directory should be excluded.

        Matching is case-insensitive.
        """

        if not path.is_dir():
            return False

        # -----------------------------------------------------
        # Global folder-name exclusion
        # -----------------------------------------------------

        if path.name.casefold() in omitted_names:
            return True

        # -----------------------------------------------------
        # Specific relative-path exclusion
        # -----------------------------------------------------

        normalized_path = get_normalized_relative_path(path)

        for omitted_path in omitted_relative_paths:

            # Exact match
            if normalized_path == omitted_path:
                return True

            # Anything nested inside an omitted folder
            if normalized_path.startswith(omitted_path + "/"):
                return True

        return False

    # ---------------------------------------------------------
    # Markdown output
    # ---------------------------------------------------------

    lines = [
        "# Folder Structure",
        "",
        f"Root: `{root.name}`",
        "",
        "```text",
        root.name,
    ]

    # ---------------------------------------------------------
    # Recursively walk directories
    # ---------------------------------------------------------

    def walk(directory, prefix=""):

        try:
            entries = list(directory.iterdir())

        except PermissionError:
            lines.append(f"{prefix}└── [Permission Denied]")
            return

        # -----------------------------------------------------
        # Remove omitted directories
        # -----------------------------------------------------

        entries = [entry for entry in entries if not should_omit(entry)]

        # -----------------------------------------------------
        # Sort:
        # 1. Directories first
        # 2. Files second
        # 3. Alphabetical, case-insensitive
        # -----------------------------------------------------

        entries.sort(
            key=lambda entry: (
                entry.is_file(),
                entry.name.casefold(),
            )
        )

        # -----------------------------------------------------
        # Generate tree structure
        # -----------------------------------------------------

        for index, entry in enumerate(entries):

            is_last = index == len(entries) - 1

            connector = "└── " if is_last else "├── "

            lines.append(f"{prefix}{connector}{entry.name}")

            if entry.is_dir():

                extension = "    " if is_last else "│   "

                walk(
                    entry,
                    prefix + extension,
                )

    # Start walking from root
    walk(root)

    lines.append("```")

    # ---------------------------------------------------------
    # Write Markdown file
    # ---------------------------------------------------------

    output_path = Path(output_file)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write("\n".join(lines))

    print(f"Markdown written to: " f"{output_path.resolve()}")


# =============================================================
# Example
# =============================================================

generate_folder_structure(
    target_folder=r"C:\PY_Programs\MSc_Project",
    output_file="agent_folder_structure.md",
    omit_folders=[
        ".venv",
        ".git",
        "__pycache__",
        "node_modules",
        "archive",
    ],
)
