"""The three track tables read as one.

Workspace v4 (2026-10-04) split the one Corpus table into Tracks, Track Sound and
Track Writing, which share `Title` and `sha256`.  A check that asks a question of
"the corpus" -- is any option unused, is any column empty on every row -- still
wants one schema and one row per track, so these two helpers put the three back
together for reading.  Nothing here writes.
"""
from __future__ import annotations

PARTS = ("tracks", "sound", "writing")


def part_ids(dbs: dict) -> list[str]:
    """The track tables a state file knows, in order.  An old state has one."""
    return [dbs[part] for part in PARTS if dbs.get(part)]


def merged_schema(api, dbs: dict) -> dict:
    props: dict = {}
    for db_id in part_ids(dbs):
        props.update(api.request("GET", f"/databases/{db_id}").get("properties") or {})
    return {"properties": props}


def merged_pages(api, dbs: dict) -> list[dict]:
    """One row per track: the properties of its page in each table, joined on sha256."""
    by_sha: dict[str, dict] = {}
    for db_id in part_ids(dbs):
        for page in api.query(db_id):
            props = page.get("properties") or {}
            rich = (props.get("sha256") or {}).get("rich_text") or []
            sha = rich[0]["text"]["content"] if rich else page["id"]
            row = by_sha.setdefault(sha, {"id": page["id"], "properties": {}})
            row["properties"].update(props)
    return list(by_sha.values())
