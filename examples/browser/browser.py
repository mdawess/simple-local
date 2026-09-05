from urllib.parse import parse_qs, quote_plus, urlparse

from playwright.sync_api import sync_playwright

# A single long-lived page the agent drives across many tool calls — this is the
# "dedicated browser instance". Everything is synchronous so the agent loop in
# agent.py stays a plain, readable for-loop.

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
CONSENT_BUTTONS = ("Accept all", "I agree", "Accept the use of cookies", "Reject all")


class Browser:
    def __init__(self, headless: bool = True):
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=headless)
        self.context = self.browser.new_context(user_agent=USER_AGENT)
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)

    def open_url(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        self.page.goto(url, wait_until="domcontentloaded")
        self._dismiss_consent()
        return f"Title: {self.page.title()}\n\n{self._visible_text()}"

    def search_web(self, query: str) -> str:
        # DuckDuckGo is a real general web search and works from an ordinary
        # machine, but it serves a bot challenge to some automated/datacenter
        # traffic. When that happens, fall back to Wikipedia so the agent always
        # gets real results to work with.
        results = self._search_duckduckgo(query)
        source = "DuckDuckGo"
        if not results:
            results = self._search_wikipedia(query)
            source = "Wikipedia (DuckDuckGo was unavailable)"
        if not results:
            return "No results found. Raw page text:\n\n" + self._visible_text()

        lines = [f"Results for {query!r} via {source}:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
        return "\n".join(lines)

    def close(self) -> None:
        self.browser.close()
        self._pw.stop()

    def __enter__(self) -> "Browser":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _search_duckduckgo(self, query: str) -> list[dict[str, str]]:
        self.page.goto(
            "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
            wait_until="domcontentloaded",
        )
        self._dismiss_consent()
        raw = self.page.evaluate(
            """
            () => {
              const out = [];
              for (const el of document.querySelectorAll('.result')) {
                const a = el.querySelector('.result__a');
                if (!a) continue;
                const snip = el.querySelector('.result__snippet');
                out.push({ title: a.innerText || '', url: a.href, snippet: (snip && snip.innerText) || '' });
              }
              return out;
            }
            """
        )
        return [_normalize(r, _unwrap_ddg(r["url"])) for r in raw[:6]]

    def _search_wikipedia(self, query: str) -> list[dict[str, str]]:
        self.page.goto(
            "https://en.wikipedia.org/w/index.php?fulltext=1&search=" + quote_plus(query),
            wait_until="domcontentloaded",
        )
        raw = self.page.evaluate(
            """
            () => {
              const out = [];
              for (const el of document.querySelectorAll('.mw-search-results li')) {
                const a = el.querySelector('.mw-search-result-heading a');
                if (!a) continue;
                const snip = el.querySelector('.searchresult');
                out.push({ title: a.innerText || '', url: a.href, snippet: (snip && snip.innerText) || '' });
              }
              return out;
            }
            """
        )
        # A single strong match redirects straight to the article (no results list).
        if not raw and "/wiki/" in self.page.url:
            return [{"title": self.page.title(), "url": self.page.url, "snippet": self._visible_text(200)}]
        return [_normalize(r, r["url"]) for r in raw[:6]]

    def _dismiss_consent(self) -> None:
        for label in CONSENT_BUTTONS:
            try:
                self.page.get_by_role("button", name=label).first.click(timeout=1500)
                return
            except Exception:
                continue

    def _visible_text(self, limit: int = 3000) -> str:
        try:
            text = self.page.inner_text("body")
        except Exception:
            text = ""
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return text[:limit]


def _normalize(result: dict, url: str) -> dict[str, str]:
    return {
        "title": " ".join((result.get("title") or "").split()),
        "url": url,
        "snippet": " ".join((result.get("snippet") or "").split())[:200],
    }


def _unwrap_ddg(url: str) -> str:
    # DuckDuckGo wraps result links in a redirect: //duckduckgo.com/l/?uddg=<real>.
    parsed = urlparse(url)
    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return url
