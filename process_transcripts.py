import re
from pathlib import Path

TRANSCRIPT_DIR = Path(".") / "transcripts"
PROCESSED_DIR = Path(".") / "processed"
MONTH_LUT = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


def read_file(file_path: Path) -> str:
    """
    Reads a transcript from a text file and returns its content.

    Args:
        file_path: Path to the transcript text file.

    Returns:
        The content of the transcript as a string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def write_lines(file_path: Path, lines: list[str]) -> None:
    """
    Writes processed transcript lines to a text file.

    Args:
        file_path: Path to the output text file.
        lines: List of lines to write to the file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def process_transcript(file_path: Path) -> list[str]:
    """
    Processes a transcript file to extract relevant contents

    Args:
        file_path: Path to the transcript text file.

    Returns:
        A list of lines from the transcript omitting the metadata section and any titles.
    """
    transcript = read_file(file_path)
    res: list[str] = []

    punctuation = {".", "!", "?", ":", ";", ",", ")"}
    for line in transcript.splitlines():
        # Skip lines that don't end with punctuation. These are things like section
        #   titles, headings, speaker, etc.
        stripped_line = line.strip()
        if not any(stripped_line.endswith(p) for p in punctuation):
            continue
        # Split the line on punctuation but don't match abbreviations and titles
        parts = re.split(
            r"(?<!\b[A-Za-z])(?<!\bDr\.)(?<=[.!?])\s+(?=[A-Z])", stripped_line
        )
        res.extend(parts)

    return res


def process_transcripts() -> None:
    """
    Processes all transcripts in the "transcripts" directory and writes
    the processed output to the "processed" directory.
    """
    PROCESSED_DIR.mkdir(exist_ok=True)
    for transcript_file in TRANSCRIPT_DIR.rglob("*.txt"):
        print(f"Processing {transcript_file.relative_to(TRANSCRIPT_DIR)}...")
        relative_path = transcript_file.relative_to(TRANSCRIPT_DIR)
        # Construct the output file name in the format:
        #   ticker-month-day-year.txt
        parts = relative_path.name.split("-")
        month = MONTH_LUT[parts[1]]
        new_name = "-".join([parts[3], parts[0], month, parts[2]]) + ".txt"
        output_file = PROCESSED_DIR / relative_path.parent / new_name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Skip transcripts that have already been processed
        if output_file.exists():
            print(f"  Skipping {transcript_file.name}, already processed.")
            continue
        # Process the transcript
        processed_lines = process_transcript(transcript_file)
        write_lines(output_file, processed_lines)
        print(f"  Wrote processed transcript to {output_file}.")


if __name__ == "__main__":
    process_transcripts()
