"""Polite HTTP client used by the scraper.

Features
- robots.txt is fetched once per host and respected
- per-host minimum delay between requests (thread-safe)
- automatic retries with exponential backoff on 429 / 5xx
- transparent gzip handling for *.xml.gz sitemaps
"""
from __future__ import annotations

import gzip
import logging
import re
import threading
import time
from urllib import robotparser
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("scraper.http")

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 DarAssistantResearchBot/1.0"
)

# Imperva / Cloudflare challenge pages (often HTTP 200 with no content)
_CHALLENGE_RE = re.compile(
    r"_Incapsula_Resource|Incapsula incident ID|<title>\s*Just a moment\.\.\.\s*</title>"
    r"|cf-chl-|challenges\.cloudflare\.com|Attention Required! \| Cloudflare",
    re.I,
)


def is_challenge(body: str | None) -> bool:
    """True if `body` is a bot-protection challenge page rather than site content."""
    return bool(body) and _CHALLENGE_RE.search(body[:20000]) is not None


class PoliteSession:
    def __init__(self, user_agent: str = DEFAULT_UA, delay: float = 0.75,
                 timeout: float = 25.0, respect_robots: bool = True):
        self.delay = delay
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._sitemaps: dict[str, list[str]] = {}
        self._last_hit: dict[str, float] = {}
        self._host_locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET", "HEAD"),
                      respect_retry_after_header=True)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        })
        self.stats = {"requests": 0, "ok": 0, "errors": 0, "blocked_by_robots": 0,
                      "circuit_open": 0, "challenged": 0}
        self._challenged_hosts: set[str] = set()
        # circuit breaker: stop hammering a host that is unreachable / blocking us
        self.max_consecutive_failures = 8
        self._fails: dict[str, int] = {}
        self._ok_hosts: set[str] = set()

    # ------------------------------------------------------------------ robots
    def _host(self, url: str) -> str:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    def _lock_for(self, host: str) -> threading.Lock:
        with self._global_lock:
            if host not in self._host_locks:
                self._host_locks[host] = threading.Lock()
            return self._host_locks[host]

    def robots(self, url: str) -> robotparser.RobotFileParser | None:
        host = self._host(url)
        if host in self._robots:
            return self._robots[host]
        rp = robotparser.RobotFileParser()
        sitemaps: list[str] = []
        try:
            r = self.session.get(f"{host}/robots.txt", timeout=self.timeout)
            if r.status_code == 200:
                lines = r.text.splitlines()
                rp.parse(lines)
                sitemaps = [ln.split(":", 1)[1].strip() for ln in lines
                            if ln.lower().startswith("sitemap:")]
            else:
                rp = None  # no robots file -> everything allowed
        except requests.RequestException as e:
            log.warning("robots.txt fetch failed for %s: %s", host, e)
            self._record(host, False)
            rp = None
        self._robots[host] = rp
        self._sitemaps[host] = sitemaps
        return rp

    def sitemaps_from_robots(self, url: str) -> list[str]:
        self.robots(url)
        return self._sitemaps.get(self._host(url), [])

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        rp = self.robots(url)
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url) and rp.can_fetch("*", url)
        except Exception:  # malformed robots
            return True

    # -------------------------------------------------------------------- GET
    def _record(self, host: str, ok: bool) -> None:
        if ok:
            self._fails[host] = 0
            self._ok_hosts.add(host)
        else:
            self._fails[host] = self._fails.get(host, 0) + 1
            if self._fails[host] == self.max_consecutive_failures:
                log.error("%s failed %d times in a row - skipping the rest of this host",
                          host, self.max_consecutive_failures)

    def circuit_open(self, url: str) -> bool:
        host = self._host(url)
        return self._fails.get(host, 0) >= self.max_consecutive_failures

    def challenged(self, url: str) -> bool:
        """True once this host has answered with a bot-protection challenge."""
        return self._host(url) in self._challenged_hosts

    def throttle(self, url: str) -> None:
        """Block until this host's minimum delay has passed (shared with the browser renderer)."""
        host = self._host(url)
        with self._lock_for(host):  # serialise per host to honour the delay
            wait = self.delay - (time.monotonic() - self._last_hit.get(host, 0))
            if wait > 0:
                time.sleep(wait)
            self._last_hit[host] = time.monotonic()

    def _flag_challenge(self, url: str, host: str) -> None:
        self.stats["challenged"] += 1
        if host not in self._challenged_hosts:
            log.warning("%s answered with a bot-protection challenge (%s) - "
                        "plain HTTP cannot read this site; use --render", host, url)
        self._challenged_hosts.add(host)
        self._record(host, False)

    def get(self, url: str, *, accept_json: bool = False) -> requests.Response | None:
        if self.circuit_open(url):
            self.stats["circuit_open"] += 1
            return None
        if not self.allowed(url):
            self.stats["blocked_by_robots"] += 1
            log.debug("robots.txt disallows %s", url)
            return None
        host = self._host(url)
        self.throttle(url)
        headers = {"Accept": "application/json"} if accept_json else None
        self.stats["requests"] += 1
        try:
            r = self.session.get(url, timeout=self.timeout, headers=headers)
        except requests.RequestException as e:
            self.stats["errors"] += 1
            self._record(host, False)
            log.warning("GET %s failed: %s", url, e)
            return None
        if r.status_code in (200, 403, 429, 503) and is_challenge(r.text):
            self._flag_challenge(url, host)
            return None
        if r.status_code != 200:
            self.stats["errors"] += 1
            # 404s are normal while crawling; only blocking/server errors trip the breaker
            self._record(host, r.status_code == 404)
            log.info("GET %s -> HTTP %s", url, r.status_code)
            return None
        self.stats["ok"] += 1
        self._record(host, True)
        return r

    def get_text(self, url: str) -> str | None:
        r = self.get(url)
        if r is None:
            return None
        if url.endswith(".gz") or r.headers.get("Content-Type", "").endswith("gzip"):
            try:
                return gzip.decompress(r.content).decode("utf-8", "replace")
            except OSError:
                pass
        r.encoding = r.encoding or r.apparent_encoding
        return r.text

    def get_json(self, url: str):
        r = self.get(url, accept_json=True)
        if r is None:
            return None
        try:
            return r.json()
        except ValueError:
            return None
