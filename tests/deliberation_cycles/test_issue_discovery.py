from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from deliberation_lake.issue_discovery.__main__ import build_discovery_id, discover, main
from deliberation_lake.issue_discovery.pipeline import (
    ALGORITHM_VERSION,
    IssueDiscovery,
    adjusted_rand_index,
)
from deliberation_lake.issue_discovery.store import DiscoveryStore

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "analysis" / "input.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def run_pipeline(payload: dict) -> dict:
    discovery_id, _ = build_discovery_id(payload)
    return discover(payload, discovery_id)


class ClusteringBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.payload = load_fixture()

    def test_six_distinct_topics_are_not_forced_together(self):
        result = run_pipeline(self.payload)
        self.assertEqual(len(result["issues"]), 6)
        titles = {issue["representative_text"] for issue in result["issues"]}
        for expected in (
            "夜间装卸噪声需要错峰",
            "公交末班时间太早请优先处理",
            "小店办证指引不清",
            "桥边照明损坏请优先处理",
            "步行入口路面破损",
            "夜班司机缺少休息点请优先处理",
        ):
            self.assertIn(expected, titles)

    def test_forwarding_burst_keeps_scale_but_not_fake_support(self):
        result = run_pipeline(self.payload)
        traffic = next(issue for issue in result["issues"] if "公交末班" in issue["title"])
        # 8 条原始提交保留传播规模。
        self.assertEqual(traffic["member_count"], 8)
        # 其中 4 条是显式转发，独立支持度只有 4，不能伪装成 8 个支持者。
        self.assertEqual(traffic["forwarded_count"], 4)
        self.assertEqual(len(traffic["forwarded_submission_ids"]), 4)
        self.assertEqual(traffic["independent_support"], 4)
        loading = next(issue for issue in result["issues"] if "夜间装卸" in issue["title"])
        self.assertEqual(loading["independent_support"], 8)
        self.assertEqual(loading["forwarded_count"], 0)

    def test_low_frequency_minority_issue_keeps_independent_agenda_slot(self):
        result = run_pipeline(self.payload)
        night_drivers = next(issue for issue in result["issues"] if "夜班司机" in issue["title"])
        self.assertEqual(night_drivers["stakeholder_coverage"]["groups_present"], ["night-driver"])
        self.assertTrue(night_drivers["meets_minimum_independent_support"])
        self.assertEqual(night_drivers["qualification"], "agenda_ready")
        self.assertEqual(result["coverage"]["stakeholder_groups_represented"],
                         result["coverage"]["stakeholder_groups_total"])

    def test_late_judgment_is_recorded_but_not_used(self):
        result = run_pipeline(self.payload)
        pending = result["judgment_ledger"]["pending_judgments"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["left"], "submission-2")
        self.assertEqual(pending[0]["status"], "pending")
        self.assertEqual(pending[0]["reason"], "available_after_as_of")
        applied_relations = {(row["left"], row["right"], row["relation"])
                             for row in result["judgment_ledger"]["applied_judgments"]}
        self.assertIn(("submission-0", "submission-6", "same"), applied_relations)
        self.assertIn(("submission-0", "submission-5", "different"), applied_relations)

    def test_different_judgment_blocks_merge_and_is_evidence(self):
        result = run_pipeline(self.payload)
        loading = next(issue for issue in result["issues"] if "夜间装卸" in issue["title"])
        self.assertTrue(any("submission-0!submission-5" in record for record in loading["blocked_merge_evidence"]))
        driver = next(issue for issue in result["issues"] if "夜班司机" in issue["title"])
        self.assertNotIn(loading["issue_id"], [driver["issue_id"]])

    def test_same_judgment_forces_merge(self):
        payload = copy.deepcopy(self.payload)
        # 把一条未来才生效的 same 改为当前生效，验证强制合并被采用。
        payload["judgments"][2]["available_at"] = "2026-08-23T00:00:00+00:00"
        result = run_pipeline(payload)
        license_issue = next(issue for issue in result["issues"] if "小店办证" in issue["title"])
        self.assertIn("submission-2", license_issue["member_submission_ids"])
        self.assertIn("submission-8", license_issue["member_submission_ids"])


class ThresholdAndWindowTests(unittest.TestCase):
    def setUp(self):
        self.payload = load_fixture()

    def test_single_voice_issue_is_retained_below_threshold_not_dropped(self):
        payload = copy.deepcopy(self.payload)
        # 仅保留一条夜班司机意见，独立支持度跌到阈值以下。
        payload["submissions"] = [
            row for row in payload["submissions"]
            if row["stakeholder"] != "night-driver" or row["submission_id"] == "submission-5"
        ]
        result = run_pipeline(payload)
        single = next(issue for issue in result["issues"] if "夜班司机" in issue["title"])
        self.assertEqual(single["independent_support"], 1)
        self.assertFalse(single["meets_minimum_independent_support"])
        self.assertEqual(single["qualification"], "candidate_below_independent_support")
        # 不静默丢弃：候选议题成员在 unassigned 中显式留痕并关联候选议题。
        row = next(item for item in result["unassigned"] if item["submission_id"] == "submission-5")
        self.assertEqual(row["reason"], "candidate_issue_below_minimum_independent_support")
        self.assertEqual(row["candidate_issue_id"], single["issue_id"])
        self.assertEqual(result["coverage"]["candidate_issues_below_support"], 1)

    def test_submission_after_cutoff_is_excluded_to_unassigned(self):
        payload = copy.deepcopy(self.payload)
        payload["submissions"].append({
            "submission_id": "submission-late",
            "text": "全新诉求：河边步道需要护栏",
            "stakeholder": "resident",
            "submitted_at": "2026-10-02T00:00:00+00:00",
            "available_at": "2026-10-02T00:00:00+00:00",
            "forwarded_from": None,
        })
        result = run_pipeline(payload)
        unassigned = next(row for row in result["unassigned"] if row["submission_id"] == "submission-late")
        self.assertEqual(unassigned["reason"], "submitted_after_discovery_cutoff")
        self.assertNotIn("submission-late", [sid for issue in result["issues"] for sid in issue["member_submission_ids"]])

    def test_unmatched_opinion_goes_to_unassigned_with_nearest_issue(self):
        payload = copy.deepcopy(self.payload)
        payload["submissions"].append({
            "submission_id": "submission-novel",
            "text": "河畔步道护栏加装诉求",
            "stakeholder": "resident",
            "submitted_at": "2026-09-15T00:00:00+00:00",
            "available_at": "2026-09-15T00:00:00+00:00",
            "forwarded_from": None,
        })
        result = run_pipeline(payload)
        row = next(item for item in result["unassigned"] if item["submission_id"] == "submission-novel")
        # 全新议题只有一个独立声音，作为候选议题留痕而非塞进无关议题。
        self.assertEqual(row["reason"], "candidate_issue_below_minimum_independent_support")
        self.assertIsNotNone(row["candidate_issue_id"])
        candidate = next(issue for issue in result["issues"] if issue["issue_id"] == row["candidate_issue_id"])
        self.assertEqual(candidate["member_submission_ids"], ["submission-novel"])
        self.assertFalse(candidate["meets_minimum_independent_support"])

    def test_topics_never_exceed_max_topics(self):
        result = run_pipeline(self.payload)
        chosen = result["stability"]["selection"]["chosen_cluster_count"]
        self.assertLessEqual(chosen, self.payload["discovery"]["max_topics"])
        self.assertGreaterEqual(chosen, 1)


class LineageAndCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.payload = load_fixture()

    def test_previous_issue_anchors_to_current_issue_with_commitment(self):
        result = run_pipeline(self.payload)
        relation = next(row for row in result["lineage"]["relations"] if row["previous_issue_id"] == "issue-traffic")
        traffic = next(issue for issue in result["issues"] if "公交末班" in issue["title"])
        self.assertEqual(relation["current_issue_id"], traffic["issue_id"])
        self.assertEqual(relation["relation"], "anchored")
        self.assertEqual(set(relation["shared_submission_ids"]), {"submission-1", "submission-7"})
        anchor = next(row for row in result["lineage"]["anchored_commitments"] if row["commitment_id"] == "commitment-bus")
        self.assertEqual(anchor["current_issue_id"], traffic["issue_id"])

    def test_manual_merge_with_missing_reference_is_blocked_evidence(self):
        result = run_pipeline(self.payload)
        action = next(row for row in result["lineage"]["manual_actions"] if row["action_id"] == "merge-1")
        self.assertEqual(action["status"], "blocked")
        self.assertIn("issue-road", action["unresolved_issue_ids"])
        unresolved = next(row for row in result["lineage"]["unresolved_references"] if row["action_id"] == "merge-1")
        self.assertEqual(unresolved["issue_id"], "issue-road")

    def test_future_manual_action_stays_pending(self):
        payload = copy.deepcopy(self.payload)
        payload["manual_actions"][0]["available_at"] = "2026-11-20T00:00:00+00:00"
        result = run_pipeline(payload)
        action = result["lineage"]["manual_actions"][0]
        self.assertEqual(action["status"], "pending")
        self.assertEqual(action["reason"], "available_after_as_of")


class StabilityAndDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.payload = load_fixture()

    def test_stability_metrics_are_perfect_on_clean_fixture(self):
        result = run_pipeline(self.payload)
        stability = result["stability"]
        self.assertEqual(stability["bootstrap_ari_mean"], 1.0)
        self.assertEqual(stability["naive_equal_weight_comparison"]["adjusted_rand_index_vs_base"], 1.0)
        self.assertEqual(stability["naive_equal_weight_comparison"]["base_cluster_preservation"], 1.0)
        self.assertEqual(stability["leave_one_root_out_preservation_min"], 1.0)

    def test_adjusted_rand_index_basics(self):
        self.assertEqual(adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertLess(adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]), 1.0)
        self.assertEqual(adjusted_rand_index([0], [0]), 0.0)

    def test_discovery_id_is_input_fingerprint_and_stable(self):
        first_id, first_fp = build_discovery_id(self.payload)
        second_id, second_fp = build_discovery_id(copy.deepcopy(self.payload))
        self.assertEqual(first_id, second_id)
        self.assertEqual(first_fp, second_fp)
        self.assertTrue(first_id.startswith("discovery-"))
        changed = copy.deepcopy(self.payload)
        changed["submissions"][0]["text"] = "夜间装卸噪声需要错峰并加设隔音屏"
        third_id, _ = build_discovery_id(changed)
        self.assertNotEqual(first_id, third_id)

    def test_deterministic_output_across_processes(self):
        first = run_pipeline(self.payload)
        second = run_pipeline(copy.deepcopy(self.payload))
        self.assertEqual(json.dumps(first, sort_keys=True, ensure_ascii=False),
                         json.dumps(second, sort_keys=True, ensure_ascii=False))


class PersistenceReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "commitments.sqlite3"
        self.first_output = Path(self.temp.name) / "first.json"
        self.replay_output = Path(self.temp.name) / "replay.json"

    def test_cli_replay_returns_same_discovery_id_and_content(self):
        payload = load_fixture()
        main(["--input", str(FIXTURE), "--database", str(self.database), "--output", str(self.first_output)])
        main(["--input", str(FIXTURE), "--database", str(self.database), "--output", str(self.replay_output)])
        first = json.loads(self.first_output.read_text())
        replay = json.loads(self.replay_output.read_text())
        self.assertFalse(first["replay"])
        self.assertTrue(replay["replay"])
        self.assertEqual(first["discovery_id"], replay["discovery_id"])
        replay.pop("replay")
        first.pop("replay")
        self.assertEqual(json.dumps(first, sort_keys=True, ensure_ascii=False),
                         json.dumps(replay, sort_keys=True, ensure_ascii=False))

    def test_database_persists_issues_lineage_and_events_isolated_from_core_tables(self):
        main(["--input", str(FIXTURE), "--database", str(self.database), "--output", str(self.first_output)])
        connection = sqlite3.connect(self.database)
        try:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            self.assertIn("issue_discovery_runs", tables)
            self.assertIn("issue_discovery_issues", tables)
            self.assertIn("issue_discovery_lineage", tables)
            self.assertIn("issue_discovery_events", tables)
            issue_count = connection.execute("SELECT COUNT(*) FROM issue_discovery_issues").fetchone()[0]
            self.assertEqual(issue_count, 6)
            lineage_count = connection.execute("SELECT COUNT(*) FROM issue_discovery_lineage").fetchone()[0]
            self.assertGreaterEqual(lineage_count, 3)
            event_count = connection.execute("SELECT COUNT(*) FROM issue_discovery_events").fetchone()[0]
            # 首次提交 + 重放各至少一条事件。
            self.assertGreaterEqual(event_count, 1)
        finally:
            connection.close()

    def test_store_recall_roundtrip(self):
        result = run_pipeline(load_fixture())
        discovery_id = result["discovery_id"]
        store = DiscoveryStore(self.database)
        self.assertTrue(store.save(discovery_id, "fingerprint-demo", result))
        self.assertFalse(store.save(discovery_id, "fingerprint-demo", result))
        recalled = store.recall(discovery_id)
        self.assertEqual(recalled["discovery_id"], discovery_id)
        self.assertEqual(len(recalled["issues"]), 6)


class PropagationEvidenceTests(unittest.TestCase):
    def test_duplicate_text_and_forwarding_edges_are_listed(self):
        result = run_pipeline(load_fixture())
        evidence = result["propagation_evidence"]
        self.assertEqual(len(evidence["duplicate_text_groups"]), 6)
        for group in evidence["duplicate_text_groups"]:
            self.assertEqual(group["submission_count"], len(group["submission_ids"]))
        forwarded = {(edge["forwarded_from"], edge["submission_id"]) for edge in evidence["forwarding_edges"]}
        self.assertEqual(forwarded, {
            ("submission-1", "submission-25"),
            ("submission-1", "submission-31"),
            ("submission-1", "submission-37"),
            ("submission-1", "submission-43"),
        })

    def test_coverage_reports_raw_and_independent_counts_per_group(self):
        result = run_pipeline(load_fixture())
        groups = result["coverage"]["per_stakeholder_group"]
        self.assertEqual(groups["merchant"]["submission_count"], 16)
        self.assertEqual(groups["night-driver"]["independent_voice_count"], 8)
        self.assertTrue(all(row["represented"] for row in groups.values()))


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.payload = load_fixture()

    def test_late_available_submission_is_excluded(self):
        payload = copy.deepcopy(self.payload)
        payload["submissions"].append({
            "submission_id": "sub-future",
            "text": "未来才可见的意见",
            "stakeholder": "resident",
            "submitted_at": "2026-09-20T00:00:00+00:00",
            "available_at": "2026-11-01T00:00:00+00:00",
            "forwarded_from": None,
        })
        result = run_pipeline(payload)
        row = next(item for item in result["unassigned"] if item["submission_id"] == "sub-future")
        self.assertEqual(row["reason"], "available_after_as_of")

    def test_dangling_forward_reference_does_not_crash_and_stays_independent(self):
        payload = copy.deepcopy(self.payload)
        payload["submissions"].append({
            "submission_id": "sub-dangling",
            "text": "公交末班时间太早请优先处理",
            "stakeholder": "merchant",
            "submitted_at": "2026-09-21T00:00:00+00:00",
            "available_at": "2026-09-21T00:00:00+00:00",
            "forwarded_from": "submission-ghost",
        })
        result = run_pipeline(payload)
        issue = next(i for i in result["issues"] if "sub-dangling" in i["member_submission_ids"])
        self.assertIn("sub-dangling", issue["propagation_roots"])

    def test_empty_submissions_do_not_raise(self):
        payload = copy.deepcopy(self.payload)
        payload["submissions"] = []
        result = run_pipeline(payload)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["stability"]["bootstrap_runs"], 0)
        self.assertIsNone(result["stability"]["bootstrap_ari_mean"])
        self.assertEqual(result["coverage"]["submissions_in_scope"], 0)

    def test_judgment_referencing_unknown_submission_is_pending(self):
        payload = copy.deepcopy(self.payload)
        payload["judgments"].append({
            "left": "submission-0",
            "right": "submission-ghost",
            "relation": "same",
            "available_at": "2026-08-25T00:00:00+00:00",
        })
        result = run_pipeline(payload)
        pending = next(j for j in result["judgment_ledger"]["pending_judgments"]
                       if j["right"] == "submission-ghost")
        self.assertEqual(pending["reason"], "reference_outside_analysis_window")


if __name__ == "__main__":
    unittest.main()
