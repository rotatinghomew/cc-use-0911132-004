"""命令行入口：

    python3 -m deliberation_lake.issue_discovery \
        --input <opinions_json> --database <commitment_db> --output <issues_json>

同一输入（含算法版本）产生同一 discovery_id；重复执行走数据库重放，
输出字节级一致，保证被承诺引用的议题含义不会因重跑而漂移。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from deliberation_lake.issue_discovery.pipeline import ALGORITHM_VERSION, IssueDiscovery, fingerprint
from deliberation_lake.issue_discovery.store import DiscoveryStore

DISCOVERY_VERSION = "d1"


def build_discovery_id(payload: Mapping[str, Any]) -> tuple[str, str]:
    """发现身份只取决于输入事实与算法版本，与运行时刻、机器无关。"""
    identity_material = {
        "algorithm_version": ALGORITHM_VERSION,
        "discovery_version": DISCOVERY_VERSION,
        "schema_version": payload.get("schema_version"),
        "as_of": payload.get("as_of"),
        "source_versions": payload.get("source_versions"),
        "discovery": payload.get("discovery"),
        "submissions": payload.get("submissions"),
        "judgments": payload.get("judgments"),
        "previous_issues": payload.get("previous_issues"),
        "manual_actions": payload.get("manual_actions"),
    }
    input_fingerprint = fingerprint(identity_material)
    discovery_id = "discovery-" + fingerprint(identity_material, 16)
    return discovery_id, input_fingerprint


def discover(payload: Mapping[str, Any], discovery_id: str) -> dict[str, Any]:
    outcome = IssueDiscovery(payload).run()
    result = {
        "discovery_id": discovery_id,
        "discovery_version": DISCOVERY_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "schema_version": payload.get("schema_version"),
        "as_of": payload.get("as_of"),
        "cutoff": (payload.get("discovery") or {}).get("cutoff"),
        "source_versions": payload.get("source_versions"),
        "baseline_case_id": payload.get("baseline_case_id"),
        "issues": outcome["issues"],
        "unassigned": outcome["unassigned"],
        "coverage": outcome["coverage"],
        "stability": outcome["stability"],
        "lineage": outcome["lineage"],
        "judgment_ledger": outcome["judgment_ledger"],
        "propagation_evidence": {
            "duplicate_text_groups": outcome["propagation"]["duplicate_text_groups"],
            "forwarding_edges": [
                {"forwarded_from": left, "submission_id": right}
                for left, right in outcome["propagation"]["forwarding_edges"]
            ],
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地协商意见议题发现（不依赖外部模型）")
    parser.add_argument("--input", required=True, help="合成意见 JSON 路径")
    parser.add_argument("--database", required=True, help="承诺/发现 SQLite 数据库路径")
    parser.add_argument("--output", required=True, help="议题发现结果 JSON 输出路径")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    discovery_id, input_fingerprint = build_discovery_id(payload)
    store = DiscoveryStore(args.database)

    replayed = store.recall(discovery_id)
    if replayed is not None:
        result = replayed
        is_replay = True
        store.record_event(discovery_id, "discovery_replayed", {"input": str(args.input)})
    else:
        result = discover(payload, discovery_id)
        stored = store.save(discovery_id, input_fingerprint, result)
        if not stored:  # 并发场景下被其他进程抢先落库。
            replayed = store.recall(discovery_id)
            assert replayed is not None
            result = replayed
            is_replay = True
        else:
            is_replay = False
            store.record_event(discovery_id, "discovery_committed", {"input": str(args.input)})

    # replay 只描述本次调用方式，不进入落库内容，因此首次与重放的核心结果一致。
    result["replay"] = is_replay
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
