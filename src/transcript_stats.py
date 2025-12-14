import pandas as pd

from master_dictionary import load_masterdictionary
from process_transcripts import BASE_DIR, PROCESSED_DIR
from risk_synonyms import RISK_SYNONYMS as RS


def create_transcripts_dataframe() -> pd.DataFrame:
    """
    Create a pandas dataframe from the transcripts dictionary.

    Returns:
        pd.DataFrame: Dataframe with columns: ticker, year, month, day, filename
    """
    rows = []

    for transcript_file in PROCESSED_DIR.rglob("*.txt"):
        ticker, year, month, day, id = transcript_file.stem.split("-")

        rows.append(
            {
                "ticker": ticker,
                "year": year,
                "month": month,
                "day": day,
                "filename": transcript_file,
            }
        )

    return pd.DataFrame(rows)


def create_regions_dict(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Create a dictionary mapping regions to lists of company tickers.

    Returns:
        dict[str, list[str]]: Dictionary with keys "UK", "US", and "Global" mapping to lists of tickers.
    """
    tickers_by_region = {"UK": [], "US": [], "Global": []}

    for ticker in df["ticker"].unique():
        # UK tickers end with .L (London Stock Exchange)
        if ticker.endswith(".L"):
            tickers_by_region["UK"].append(ticker)
            continue
        # US tickers end with .N, .O, or .OQ
        us_tickers = [".N", ".O", ".OQ"]
        for suffix in us_tickers:
            if ticker.endswith(suffix):
                tickers_by_region["US"].append(ticker)
                break
        # If not UK or US, classify as Global
        else:
            tickers_by_region["Global"].append(ticker)

    return tickers_by_region


def calculate_brexit_exposure(transcript: str) -> float:
    """
    Calculate Brexit exposure for a given transcript.

    Brexit exposure is defined as the number of times "brexit" appears divided by total word count.

    Args:
        transcript (str): The transcript text

    Returns:
        float: Brexit exposure value (between 0 and 1)
    """
    transcript_lower = transcript.lower()
    brexit_count = transcript_lower.count("brexit")
    words = transcript_lower.split()
    total_words = len(words)

    if total_words == 0:
        return 0.0

    return brexit_count / total_words


def calculate_brexit_risk(transcript: str) -> float:
    """
    Calculate Brexit risk for a given transcript.

    Brexit risk is defined as the number of times "brexit" appears within a 10-word
    neighborhood of any risk synonym word, divided by total word count.

    Args:
        transcript (str): The transcript text

    Returns:
        float: Brexit risk value
    """
    transcript_words = transcript.lower().split()
    total_words = len(transcript_words)

    # Avoid division by zero
    if total_words == 0:
        return 0.0

    risk_synonyms_lower = [rs.lower() for rs in RS]

    # Track positions of risk synonym words
    risk_positions = []
    for i, word in enumerate(transcript_words):
        if word in risk_synonyms_lower:
            risk_positions.append(i)

    # Count brexit occurrences within 10-word neighborhood of risk words
    brexit_risk_count = 0

    for i, word in enumerate(transcript_words):
        if word == "brexit":
            # Check if this brexit is within 10 words of any risk synonym
            for risk_pos in risk_positions:
                if abs(i - risk_pos) < 10:
                    brexit_risk_count += 1

    return brexit_risk_count


def calculate_brexit_sentiment(transcript: str, md) -> float:
    """
    Calculate Brexit sentiment for a given transcript.

    Brexit sentiment is defined as the sum of sentiment scores in 10-word neighborhoods
    around "brexit" mentions, divided by total word count. Positive words add +1,
    negative words add -1.

    Args:
        transcript (str): The transcript text
        md: The master dictionary for sentiment analysis

    Returns:
        float: Brexit sentiment value
    """
    transcript_words = transcript.split()
    transcript_words_lower = [word.lower() for word in transcript_words]
    total_words = len(transcript_words)

    if total_words == 0:
        return 0.0

    # Find all brexit positions
    brexit_positions = []
    for i, word in enumerate(transcript_words_lower):
        if word == "brexit":
            brexit_positions.append(i)

    # Calculate sentiment for each brexit occurrence
    brexit_sentiment = 0

    for brexit_pos in brexit_positions:
        # Define 10-word neighborhood (10 words before and after)
        start = max(0, brexit_pos - 10)
        end = min(total_words, brexit_pos + 11)

        # Check each word in neighborhood for sentiment
        for i in range(start, end):
            word_upper = transcript_words[i].upper()

            # Check if word exists in master dictionary
            if word_upper in md:
                word_entry = md[word_upper]
                # Add +1 for positive words
                if word_entry.positive > 0:
                    brexit_sentiment += 1
                # Add -1 for negative words
                if word_entry.negative > 0:
                    brexit_sentiment -= 1

    return brexit_sentiment


def calculate_non_brexit_risk(transcript: str, brexit_risk: float) -> float:
    """
    Calculate non-Brexit risk for a given transcript.

    Non-Brexit risk is the total number of risk synonym occurrences minus Brexit risk.

    Args:
        transcript (str): The transcript text

    Returns:
        float: Non-Brexit risk value
    """
    transcript_words = transcript.lower().split()
    total_words = len(transcript_words)

    if total_words == 0:
        return 0.0

    risk_synonyms_lower = [rs.lower() for rs in RS]

    # Count total risk synonym occurrences
    total_risk_count = 0
    for word in transcript_words:
        if word in risk_synonyms_lower:
            total_risk_count += 1

    return total_risk_count - brexit_risk


def calculate_non_brexit_sentiment(
    transcript: str, brexit_sentiment: float, md
) -> float:
    """
    Calculate non-Brexit sentiment for a given transcript.

    Non-Brexit sentiment is the total sentiment of the transcript minus Brexit sentiment.
    Total sentiment is calculated by adding +1 for positive words and -1 for negative words.

    Args:
        transcript (str): The transcript text
        brexit_sentiment (float): The Brexit sentiment value to subtract
        md: The master dictionary for sentiment analysis

    Returns:
        float: Non-Brexit sentiment value
    """
    transcript_words = transcript.upper().split()
    total_sentiment = 0

    for word in transcript_words:
        if word in md:
            word_entry = md[word]
            # Add +1 for positive words
            if word_entry.positive > 0:
                total_sentiment += 1
            # Add -1 for negative words
            if word_entry.negative > 0:
                total_sentiment -= 1

    return total_sentiment - brexit_sentiment


def get_transcript_stats() -> pd.DataFrame:
    # Create the initial dataframe
    df = create_transcripts_dataframe()
    # Load Loughran-McDonald Master Dictionary
    file_name = "Loughran-McDonald_MasterDictionary_1993-2024.csv"
    dir_name = "loughran_mcdonald_dictionary"
    path = BASE_DIR / dir_name / file_name
    md = load_masterdictionary(path)

    # Create Brexit-related columns
    def calc_stats(row):
        transcript = open(row["filename"], "r").read()
        return (
            calculate_brexit_exposure(transcript),
            calculate_brexit_risk(transcript),
            calculate_brexit_sentiment(transcript, md),
        )

    df["BrexitExposure"], df["BrexitRisk"], df["BrexitSentiment"] = zip(
        *df.apply(calc_stats, axis=1)
    )
    # Standardize BrexitRisk and BrexitSentiment by UK-based companies
    tickers_by_region = create_regions_dict(df)
    df_uk = df[df["ticker"].isin(tickers_by_region["UK"])]
    df["BrexitRisk"] = df["BrexitRisk"] / df_uk["BrexitRisk"].mean()
    df["BrexitSentiment"] = df["BrexitSentiment"] / df_uk["BrexitSentiment"].mean()

    # Create Non-Brexit-related columns
    def calc_non_brexit_stats(row):
        transcript = open(row["filename"], "r").read()
        return (
            calculate_non_brexit_risk(transcript, row["BrexitRisk"]),
            calculate_non_brexit_sentiment(transcript, row["BrexitSentiment"], md),
        )

    df["NonBrexitRisk"], df["NonBrexitSentiment"] = zip(
        *df.apply(calc_non_brexit_stats, axis=1)
    )

    return df


if __name__ == "__main__":
    df_stats = get_transcript_stats()
    print(df_stats.head())
