"""A polite, cached HTTP client for the metadata providers.

Three rules the public music databases all ask for and mtx honours here:

  * one identifiable User-Agent, so an operator can find out who is calling;
  * a per-host minimum interval, because MusicBrainz enforces one request per
    second and will start answering 503 to a caller that ignores it;
  * a disk cache, so re-running `mtx enrich` over a folder that was already
    enriched costs nothing and works with the network unplugged.

Stdlib only.  The online module is optional and must not drag a dependency
into a package whose point is reproducible local measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

# Seconds a host must be left alone between calls.  MusicBrainz publishes the
# 1 req/s figure; the rest are courtesy values, chosen low enough that a
# 64-track corpus still finishes in a couple of minutes.
MIN_INTERVAL = {
    "musicbrainz.org": 1.1,
    "api.discogs.com": 1.1,
    "itunes.apple.com": 0.35,
    "ws.audioscrobbler.com": 0.25,
    "api.deezer.com": 0.15,
    "api.listenbrainz.org": 0.25,
}
DEFAULT_INTERVAL = 0.5

# 429 and 503 are the two "come back later" answers these APIs use.
RETRY_STATUS = (429, 500, 502, 503, 504)


class Client:
    """Rate-limited, disk-cached JSON GET.

    Every response is stored under a hash of the full URL.  A cached entry is
    returned without touching the network unless `refresh` is set, which makes
    a second enrich pass free and keeps a corpus reproducible: the same cache
    yields the same `online` section months later, after the upstream data has
    moved on.
    """

    def __init__(self, cache_dir: str | None, user_agent: str,
                 log: Callable[[str], None] | None = None,
                 offline: bool = False, refresh: bool = False,
                 timeout: float = 30.0, max_retries: int = 4) -> None:
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        self.log = log or (lambda _m: None)
        self.offline = offline
        self.refresh = refresh
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_call: dict[str, float] = {}
        self.stats = {"hit": 0, "miss": 0, "error": 0, "skipped": 0}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, url: str) -> str | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.json")

    def _read_cache(self, url: str) -> dict[str, Any] | None:
        path = self._cache_path(url)
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _write_cache(self, url: str, entry: dict[str, Any]) -> None:
        path = self._cache_path(url)
        if not path:
            return
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            pass  # a cache that cannot be written is a slow run, not a failure

    # -- fetch ---------------------------------------------------------------

    def _wait(self, host: str) -> None:
        interval = MIN_INTERVAL.get(host, DEFAULT_INTERVAL)
        elapsed = time.monotonic() - self._last_call.get(host, 0.0)
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call[host] = time.monotonic()

    def get_json(self, url: str, headers: dict[str, str] | None = None
                 ) -> tuple[Any | None, str | None]:
        """Return (body, error).  Exactly one of the two is None."""
        if not self.refresh:
            cached = self._read_cache(url)
            if cached is not None:
                self.stats["hit"] += 1
                return cached.get("body"), cached.get("error")
        if self.offline:
            self.stats["skipped"] += 1
            return None, "offline and not cached"

        host = urllib.parse.urlparse(url).netloc
        request_headers = {"User-Agent": self.user_agent,
                           "Accept": "application/json"}
        request_headers.update(headers or {})

        error: str | None = None
        for attempt in range(self.max_retries):
            self._wait(host)
            req = urllib.request.Request(url, headers=request_headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                body = json.loads(raw.decode("utf-8", "replace")) if raw else None
                self.stats["miss"] += 1
                self._write_cache(url, {"url": url, "body": body, "error": None,
                                        "fetched_utc": _utc()})
                return body, None
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    # A miss is a real answer and worth caching: most ISRCs that
                    # one provider does not know it will still not know tomorrow.
                    error = "404 not found"
                    self.stats["miss"] += 1
                    self._write_cache(url, {"url": url, "body": None,
                                            "error": error, "fetched_utc": _utc()})
                    return None, error
                error = f"HTTP {exc.code}"
                if exc.code in RETRY_STATUS:
                    delay = _retry_after(exc) or (1.5 * (2 ** attempt))
                    self.log(f"{host} {exc.code}, retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                error = f"network error: {exc}"
                time.sleep(1.0 * (2 ** attempt))
            except ValueError as exc:
                error = f"bad JSON: {exc}"
                break

        self.stats["error"] += 1
        # Transport failures are deliberately not cached: unlike a 404 they say
        # nothing about the data, and caching them would poison the next run.
        return None, error


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    try:
        return float(exc.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        return None


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_url(base: str, **params: Any) -> str:
    clean = {k: v for k, v in params.items() if v not in (None, "")}
    return base + ("&" if "?" in base else "?") + urllib.parse.urlencode(clean)
