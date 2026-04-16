from pathlib import Path
import json

from tools.wikipedia_api import fetch_articles_for_topics

RAW_OUTPUT_DIR = Path("data/raw/wikipedia")
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_safe_filename(title: str) -> str:
    """
    Turn a Wikipedia page title into a safe filename.
    """
    safe = title.replace("/", "-").replace("\\", "-")
    safe = safe.replace(" ", "_").replace(":", "-")
    safe = safe.replace("?", "").replace('"', "")
    safe = safe.replace("<", "").replace(">", "").replace("|", "")
    return safe


def save_article_file(title: str, text: str) -> str:
    """
    Save one article to a text file.
    
    Returns:
        The filename used
    """
    filename = f"{make_safe_filename(title)}.txt"
    filepath = RAW_OUTPUT_DIR / filename
    filepath.write_text(text, encoding="utf-8")
    return filename


def save_manifest(records: list[dict]) -> None:
    """
    Save metadata about downloaded articles.
    """
    manifest_path = RAW_OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def save_combined_corpus(articles: list[dict]) -> None:
    """
    Save all article text into one combined training file.
    """
    combined_path = RAW_OUTPUT_DIR / "wikipedia_corpus.txt"

    with combined_path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(article["title"])
            f.write("\n\n")
            f.write(article["text"])
            f.write("\n\n")
            f.write("=" * 80)
            f.write("\n\n")


def main() -> None:
    topics = [
        "pyramid construction",
        "Egyptian pyramids",
        "pyramid engineering",
        "ancient Egyptian architecture",
        "Great Pyramid of Giza construction",
    ]

    articles = fetch_articles_for_topics(topics, limit=25)

    manifest_records = []

    for article in articles:
        filename = save_article_file(article["title"], article["text"])
        manifest_records.append({
            "title": article["title"],
            "filename": filename,
            "characters": len(article["text"]),
        })

    save_manifest(manifest_records)
    save_combined_corpus(articles)

    print(f"Saved {len(articles)} articles to {RAW_OUTPUT_DIR}")
    print(f"Combined corpus file: {RAW_OUTPUT_DIR / 'wikipedia_corpus.txt'}")
    print(f"Manifest file: {RAW_OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()