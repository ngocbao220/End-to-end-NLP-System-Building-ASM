from __future__ import annotations

import argparse
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"p", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        text = " ".join(part.strip() for part in self.parts if part.strip())
        return re.sub(r"\s+", " ", text).strip()


def fetch_text(url: str, timeout: int = 20) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 educational-rag-bot"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        raw = response.read()
    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        return ""
    html = raw.decode("utf-8", errors="ignore")
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape configured public HTML pages into JSONL documents.")
    parser.add_argument("--urls", type=Path, default=Path("data/raw/source_urls.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/scraped_documents.jsonl"))
    args = parser.parse_args()

    urls = [line.strip() for line in args.urls.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for url in urls:
            text = fetch_text(url)
            if not text:
                continue
            title = urlparse(url).path.strip("/").split("/")[-1] or urlparse(url).netloc
            record = {"url": url, "title": title, "text": text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"scraped {url} ({len(text)} chars)")


if __name__ == "__main__":
    main()
