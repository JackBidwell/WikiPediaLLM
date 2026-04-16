import requests
from typing import List, Dict

BASE_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "WikipediaLLM/0.1 (personal learning project)"
}


def search_wikipedia(query: str, limit: int = 10) -> List[Dict]:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "srprop": "snippet",
    }

    response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()

    data = response.json()
    return data.get("query", {}).get("search", [])


def fetch_article_extract(title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
    }

    response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    if not pages:
        return ""

    page = next(iter(pages.values()))
    return page.get("extract", "")


def fetch_articles_for_query(query: str, limit: int = 10) -> List[Dict[str, str]]:
    search_results = search_wikipedia(query, limit=limit)
    articles = []
    seen_titles = set()

    for result in search_results:
        title = result.get("title", "")

        if not title or title in seen_titles:
            continue

        seen_titles.add(title)
        text = fetch_article_extract(title)

        if text.strip():
            articles.append({
                "title": title,
                "text": text,
            })

    return articles


def fetch_articles_for_topics(topics: List[str], limit: int = 50) -> List[Dict[str, str]]:
    articles = []
    seen_titles = set()

    per_topic_limit = max(1, limit // max(1, len(topics)))

    for topic in topics:
        search_results = search_wikipedia(topic, limit=per_topic_limit)

        for result in search_results:
            title = result.get("title", "")

            if not title or title in seen_titles:
                continue

            seen_titles.add(title)
            text = fetch_article_extract(title)

            if text.strip():
                articles.append({
                    "title": title,
                    "text": text,
                })

            if len(articles) >= limit:
                return articles

    return articles


if __name__ == "__main__":
    test_articles = fetch_articles_for_query("construction of the pyramids", limit=5)

    print(f"Retrieved {len(test_articles)} articles\n")

    for article in test_articles:
        print(article["title"])
        print(article["text"][:300])
        print("-" * 60)