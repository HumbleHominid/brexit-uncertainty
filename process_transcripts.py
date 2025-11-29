import re
from pathlib import Path


def read_transcript(file_path: Path) -> str:
    """
    Reads a transcript from a text file and returns its content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def write_processed_transcript(file_path: Path, lines: list[str]) -> None:
    """
    Writes processed transcript lines to a text file.

    Args:
        file_path: Path to the output text file.
        lines: List of lines to write to the file.
    """
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
    transcript = read_transcript(file_path)
    res: list[str] = []

    punctuation = {".", "!", "?", ":", ";", ","}
    for line in transcript.splitlines():
        # Skip lines that don't end with punctuation. These are things like section
        #   titles, headings, speaker, etc.
        if not any(line.strip().endswith(p) for p in punctuation):
            continue
        # Split the line on punctuation but don't match abbreviations and titles
        parts = re.split(r"(?<!\b[A-Za-z])(?<!\bDr\.)(?<=[.!?])\s+(?=[A-Z])", line)
        res.extend(parts)

    return res


def process_transcripts() -> None:
    """
    Processes all transcripts in the "transcripts" directory and writes
    the processed output to the "processed" directory.
    """
    # process all transcripts in the "transcripts" directory writing output to the "processed" directory
    input_dir = Path(".") / "transcripts"
    output_dir = Path(".") / "processed"
    output_dir.mkdir(exist_ok=True)

    for transcript_file in input_dir.glob("*.txt"):
        print(f"Processing {transcript_file.name}...")
        # Skip transcripts that have already been processed
        output_file = output_dir / transcript_file.name
        if output_file.exists():
            print(f"  Skipping {transcript_file.name}, already processed.")
            continue
        # Process the transcript
        processed_lines = process_transcript(transcript_file)
        write_processed_transcript(output_file, processed_lines)
        print(f"  Wrote processed transcript to {output_file}.")


if __name__ == "__main__":
    process_transcripts()
