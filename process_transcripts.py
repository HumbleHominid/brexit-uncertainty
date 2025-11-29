import re
from pathlib import Path


def read_transcript(file_path: Path) -> str:
    """
    Reads a transcript from a text file and returns its content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def extract_metadata(transcript: str) -> dict:
    """
    Extracts participant metadata from an earnings-call transcript.

    The function looks for lines in the transcript that follow a common
    pattern used in earnings call docs, e.g.:

        James Friedland - Senior Director of Investor Relations

    It supports common dash characters ("-", "–", "—") and returns a
    dictionary mapping participant names to their titles.

    Args:
        transcript: Full transcript text as a string.

    Returns:
        A dict where keys are participant names (str) and values are their
        titles (str). If the same name appears multiple times, the first
        occurrence is preserved.
    """

    metadata: dict[str, str] = {}

    # Match lines like: Name - Title  (supports -, – and —)
    pattern = re.compile(r"^\s*([A-Z][A-Za-z\.'\- ]{1,200}?)\s*[-–—]\s*(.+?)\s*$")

    for line in transcript.splitlines():
        # Stop processing the metadata section when we reach the main content
        if line.startswith("Presentation"):
            break

        m = pattern.match(line)
        if not m:
            continue

        name = m.group(1).strip()
        title = m.group(2).strip()

        # Basic sanity: discard lines where title looks like a company or
        # short one-word tokens that are unlikely to be roles (optional).
        if not name or not title:
            continue

        # Avoid overwriting if name already captured earlier in transcript
        if name not in metadata:
            metadata[name] = title

    return metadata


if __name__ == "__main__":
    path = Path(".") / "transcripts" / "lly-2025-q3.txt"

    transcript = read_transcript(path)
    metadata = extract_metadata(transcript)
    for name, title in metadata.items():
        print(f"{name}: {title}")
