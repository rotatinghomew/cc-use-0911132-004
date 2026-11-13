"""议题发现结果的本地持久化：幂等重放、谱系留痕、事件溯源。

与运行库共用同一个 SQLite 文件，但所有表均以 issue_discovery_ 前缀隔离，
不读取也不改动既有 runs/events/idempotency/failures 表。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from deliberation_lake.engine.state_store import stable_json
from deliberation_lake.issue_discovery.pipeline import ALGORITHM_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DiscoveryStore:
    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS issue_discovery_runs(
                    discovery_id TEXT PRIMARY KEY,
                    input_fingerprint TEXT NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    discovery_version TEXT NOT NULL,
                    source_versions TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    cutoff TEXT,
                    result TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS issue_discovery_issues(
                    discovery_id TEXT NOT NULL,
                    issue_id TEXT NOT NULL,
                    issue_key TEXT NOT NULL,
                    independent_support INTEGER NOT NULL,
                    member_count INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(discovery_id, issue_id)
                );
                CREATE TABLE IF NOT EXISTS issue_discovery_lineage(
                    discovery_id TEXT NOT NULL,
                    previous_issue_id TEXT NOT NULL,
                    current_issue_id TEXT,
                    relation TEXT NOT NULL,
                    commitment_ids TEXT NOT NULL,
                    detail TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS issue_discovery_events(
                    event_id TEXT PRIMARY KEY,
                    discovery_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS issue_discovery_lineage_previous
                    ON issue_discovery_lineage(previous_issue_id);
                """
            )
            connection.commit()
        finally:
            connection.close()

    def recall(self, discovery_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT result FROM issue_discovery_runs WHERE discovery_id=?",
                (discovery_id,),
            ).fetchone()
            return json.loads(row["result"]) if row else None
        finally:
            connection.close()

    def record_event(self, discovery_id: str, kind: str, detail: Mapping[str, Any]) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO issue_discovery_events(event_id,discovery_id,kind,detail,created_at) VALUES(?,?,?,?,?)",
                (uuid.uuid4().hex, discovery_id, kind, stable_json(detail), utc_now()),
            )
            connection.commit()
        finally:
            connection.close()

    def save(self, discovery_id: str, input_fingerprint: str, result: Mapping[str, Any]) -> bool:
        """返回 False 表示同 ID 记录已存在（重放），True 表示首次落库。"""
        connection = self._connect()
        try:
            existing = connection.execute(
                "SELECT discovery_id FROM issue_discovery_runs WHERE discovery_id=?",
                (discovery_id,),
            ).fetchone()
            if existing is not None:
                return False
            connection.execute(
                "INSERT INTO issue_discovery_runs(discovery_id,input_fingerprint,algorithm_version,"
                "discovery_version,source_versions,as_of,cutoff,result,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    discovery_id,
                    input_fingerprint,
                    str(result.get("algorithm_version", ALGORITHM_VERSION)),
                    str(result.get("discovery_version", "d1")),
                    stable_json(result.get("source_versions", {})),
                    str(result.get("as_of", "")),
                    result.get("cutoff"),
                    stable_json(result),
                    utc_now(),
                ),
            )
            for issue in result.get("issues", []):
                connection.execute(
                    "INSERT INTO issue_discovery_issues(discovery_id,issue_id,issue_key,"
                    "independent_support,member_count,payload) VALUES(?,?,?,?,?,?)",
                    (
                        discovery_id,
                        issue["issue_id"],
                        issue["issue_key"],
                        int(issue["independent_support"]),
                        int(issue["member_count"]),
                        stable_json(issue),
                    ),
                )
            for relation in result.get("lineage", {}).get("relations", []):
                connection.execute(
                    "INSERT INTO issue_discovery_lineage(discovery_id,previous_issue_id,current_issue_id,"
                    "relation,commitment_ids,detail) VALUES(?,?,?,?,?,?)",
                    (
                        discovery_id,
                        relation["previous_issue_id"],
                        relation["current_issue_id"],
                        relation["relation"],
                        "[]",
                        stable_json(relation),
                    ),
                )
            for anchored in result.get("lineage", {}).get("anchored_commitments", []):
                connection.execute(
                    "INSERT INTO issue_discovery_lineage(discovery_id,previous_issue_id,current_issue_id,"
                    "relation,commitment_ids,detail) VALUES(?,?,?,?,?,?)",
                    (
                        discovery_id,
                        anchored["previous_issue_id"],
                        anchored["current_issue_id"],
                        "commitment_anchor",
                        stable_json([anchored["commitment_id"]]),
                        stable_json(anchored),
                    ),
                )
            for unresolved in result.get("lineage", {}).get("unresolved_references", []):
                connection.execute(
                    "INSERT INTO issue_discovery_lineage(discovery_id,previous_issue_id,current_issue_id,"
                    "relation,commitment_ids,detail) VALUES(?,?,?,?,?,?)",
                    (
                        discovery_id,
                        str(unresolved.get("issue_id", unresolved.get("action_id", "unknown"))),
                        None,
                        "unresolved",
                        stable_json([]),
                        stable_json(unresolved),
                    ),
                )
            connection.execute(
                "INSERT INTO issue_discovery_events(event_id,discovery_id,kind,detail,created_at) VALUES(?,?,?,?,?)",
                (uuid.uuid4().hex, discovery_id, "discovery_persisted",
                 stable_json({"algorithm_version": result.get("algorithm_version"),
                              "issue_count": len(result.get("issues", []))}), utc_now()),
            )
            connection.commit()
            return True
        finally:
            connection.close()
