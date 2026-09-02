"""A minimal Notion client: stdlib only, rate-limited, and it retries.

`mtx`'s `online/` subpackage is stdlib-only on purpose -- a tool whose point is
reproducible local measurement should not grow a dependency tree to talk to a
web service.  The same rule applies here, so this is `urllib` and nothing else.

Notion allows roughly three requests a second per integration and answers 429
with a `Retry-After` when you exceed it.  Pushing 1,321 tracks is thousands of
requests, so the throttle is not optional: without it the run dies a third of
the way in and leaves a half-populated database.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

API = "https://api.notion.com/v1"

# Pinned deliberately.  Notion dates its API and changes response shapes
# between versions; an unpinned client breaks on someone else's release
# schedule.  2022-06-28 is the long-stable version whose database/page shapes
# this loader is written against.
NOTION_VERSION = "2022-06-28"

MIN_INTERVAL_S = 0.36        # ~2.8 req/s, just under the documented ceiling
MAX_ATTEMPTS = 6


class NotionError(RuntimeError):
    def __init__(self, status: int, body: str, path: str):
        self.status = status
        self.body = body
        super().__init__(f"{status} on {path}: {body[:400]}")


class Notion:
    def __init__(self, token: str | None = None, *, dry_run: bool = False,
                 log=None):
        self.token = token or os.environ.get("NOTION_TOKEN") or ""
        self.dry_run = dry_run
        self.requests = 0
        self._last = 0.0
        self._log = log or (lambda m: print(f"[notion] {m}", file=sys.stderr, flush=True))
        if not self.token and not dry_run:
            raise SystemExit(
                "error: no Notion token.  Set NOTION_TOKEN, or pass --dry-run "
                "to write the payloads to disk without sending them.")

    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        if self.dry_run:
            return {"id": "dry-run", "object": "dry-run", "results": []}
        url = f"{API}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle()
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Authorization", f"Bearer {self.token}")
            req.add_header("Notion-Version", NOTION_VERSION)
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    self.requests += 1
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                # 429 is expected under load and 5xx is Notion having a moment;
                # both are worth waiting out.  4xx otherwise is our own bug and
                # retrying it just makes the same mistake more slowly.
                if exc.code == 429:
                    delay = float(exc.headers.get("Retry-After") or 1.0)
                elif 500 <= exc.code < 600:
                    delay = min(2 ** attempt, 30) + random.random()
                else:
                    raise NotionError(exc.code, body, path) from None
                if attempt == MAX_ATTEMPTS:
                    raise NotionError(exc.code, body, path) from None
                self._log(f"{exc.code} on {path}; retry {attempt}/{MAX_ATTEMPTS} in {delay:.1f}s")
                time.sleep(delay)
            except urllib.error.URLError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise
                delay = min(2 ** attempt, 30) + random.random()
                self._log(f"network error ({exc.reason}); retry {attempt}/{MAX_ATTEMPTS} in {delay:.1f}s")
                time.sleep(delay)
        raise NotionError(0, "exhausted retries", path)

    # ------------------------------------------------------------------
    def create_database(self, parent_page_id: str, title: str,
                        properties: dict) -> dict:
        return self.request("POST", "/databases", {
            "parent": {"type": "page_id", "page_id": parent_page_id},
            "title": [{"type": "text", "text": {"content": title}}],
            "properties": properties,
        })

    def update_database(self, database_id: str, properties: dict) -> dict:
        return self.request("PATCH", f"/databases/{database_id}",
                            {"properties": properties})

    def create_page(self, database_id: str, properties: dict,
                    children: list | None = None) -> dict:
        payload: dict = {"parent": {"database_id": database_id},
                         "properties": properties}
        if children:
            # Notion caps page creation at 100 child blocks; the rest are
            # appended afterwards.
            payload["children"] = children[:100]
        return self.request("POST", "/pages", payload)

    def update_page(self, page_id: str, properties: dict) -> dict:
        return self.request("PATCH", f"/pages/{page_id}",
                            {"properties": properties})

    def append_blocks(self, block_id: str, children: list) -> None:
        for i in range(0, len(children), 100):
            self.request("PATCH", f"/blocks/{block_id}/children",
                         {"children": children[i:i + 100]})

    def delete_block(self, block_id: str) -> dict:
        return self.request("DELETE", f"/blocks/{block_id}")

    def children(self, block_id: str) -> list:
        out, cursor = [], None
        while True:
            path = f"/blocks/{block_id}/children?page_size=100"
            if cursor:
                path += f"&start_cursor={cursor}"
            res = self.request("GET", path)
            out.extend(res.get("results") or [])
            if not res.get("has_more"):
                return out
            cursor = res.get("next_cursor")

    def query(self, database_id: str, filter_: dict | None = None,
              page_size: int = 100) -> list:
        out, cursor = [], None
        while True:
            payload: dict = {"page_size": page_size}
            if filter_:
                payload["filter"] = filter_
            if cursor:
                payload["start_cursor"] = cursor
            res = self.request("POST", f"/databases/{database_id}/query", payload)
            out.extend(res.get("results") or [])
            if not res.get("has_more"):
                return out
            cursor = res.get("next_cursor")

    def find_databases(self, parent_page_id: str) -> dict[str, str]:
        """`{title: id}` for the child databases of a page."""
        found = {}
        for block in self.children(parent_page_id):
            if block.get("type") != "child_database":
                continue
            title = (block.get("child_database") or {}).get("title") or ""
            if title:
                found[title] = block["id"]
        return found
