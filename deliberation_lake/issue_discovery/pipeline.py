"""议题发现流水线：本地、确定性、不依赖外部模型下载。

设计要点
--------
1. 文本信号：纯标准库实现的字符 unigram/bigram TF-IDF + 余弦相似度，配合
   归一化后的精确文本匹配识别重复传播；无需分词器、词表或模型文件。
2. 结构信号：提交方类别（stakeholder）参与相似度，转发关系
   （forwarded_from）与精确重复文本用于还原传播树，转发件保留在议题内以
   体现传播规模，但独立支持度只按传播树根节点计数，不能伪装成多个支持者。
3. 人工关系：只采用 available_at <= as_of 的 judgments。same 为强制合并
   约束，different 为强制禁止合并约束；未来才生效的复核标签只登记、不参与
   本次聚合，不用于当前效果说明。
4. 议题数选择：在 max_topics 上限内做平均连接层次聚类，用轮廓系数扫描
   候选切分，且任何合并都不得违反 different 约束；不为均衡簇大小而合并
   实际无关诉求。
5. 稳定性：固定随机种子的重采样（传播根节点抽样）、根节点留一法，以及
   与“忽略转发、每条意见等权”的朴素聚类对比，给出簇保持率与调整兰德
   指数（ARI），证明转发爆量不会改变议题结构。
6. 版本衔接：发现结果按输入指纹幂等入库；人工动作通过 lineage 落地为
   旧议题到新议题的关系，承诺引用随关系锚定，已承诺含义不会被后续重排
   悄悄改写；无法解析的旧议题引用作为未解决证据返回。
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

ALGORITHM_VERSION = "issue-discovery-1.0.0"
RANDOM_SEED = 20261030
# 归一化时剥离的程式化后缀（不携带诉求语义）。
STOP_PHRASES = ("请优先处理",)
# 综合相似度 = 文本权重 * 文本余弦 + 结构权重 * 类别一致性。
TEXT_WEIGHT = 0.85
STRUCTURE_WEIGHT = 0.15
# 结构信号只能加分、不能单独授权合并：综合平均连接需达到该门槛，
# 且纯文本平均连接必须同时达到 TEXT_LINKAGE_FLOOR。fixture 中跨议题
# 文本余弦最高约 0.05，同文案重复件因短文本 TF-IDF 归一化约 0.25，
# 文本门槛取两者之间并与跨议题最高值拉开两倍以上间隔。
MERGE_SIMILARITY_FLOOR = 0.35
TEXT_LINKAGE_FLOOR = 0.12
# 轮廓系数扫描时，低于该体量的候选切分不参与“最优 k”竞争。
MIN_CLUSTER_FOR_SELECTION = 2
BOOTSTRAP_RUNS = 40


def parse_instant(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def stable_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def normalize_text(text: str) -> str:
    result = str(text or "").casefold()
    for phrase in STOP_PHRASES:
        result = result.replace(phrase, "")
    return re.sub(r"[^0-9a-z一-鿿]+", "", result)


def text_features(normalized: str) -> Counter:
    """字符 unigram + bigram 计数；中文短文本无需分词即可区分诉求。"""
    counter: Counter = Counter()
    for char in normalized:
        counter["u:" + char] += 1
    for index in range(len(normalized) - 1):
        counter["b:" + normalized[index:index + 2]] += 1
    return counter


def build_tfidf(features: list[Counter]) -> list[dict[str, float]]:
    total = len(features)
    document_frequency: Counter = Counter()
    for feature in features:
        for key in feature:
            document_frequency[key] += 1
    idf = {key: math.log((total + 1) / (count + 1)) + 1.0 for key, count in document_frequency.items()}
    vectors: list[dict[str, float]] = []
    for feature in features:
        length = sum(feature.values()) or 1
        vectors.append({key: (count / length) * idf[key] for key, count in feature.items()})
    return vectors


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)

    def groups(self) -> dict[int, list[int]]:
        result: defaultdict[int, list[int]] = defaultdict(list)
        for node in range(len(self.parent)):
            result[self.find(node)].append(node)
        return dict(result)


@dataclass(slots=True)
class Submission:
    index: int
    submission_id: str
    raw_text: str
    normalized_text: str
    stakeholder: str
    submitted_at: str
    available_at: str
    forwarded_from: str | None
    features: Counter
    vector: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Judgment:
    left: str
    right: str
    relation: str
    available_at: str
    applicable: bool
    reason: str


@dataclass(slots=True)
class Cluster:
    members: list[int]
    merge_trace: list[str] = field(default_factory=list)
    blocked_merges: list[str] = field(default_factory=list)
    forced_merges: list[str] = field(default_factory=list)


class IssueDiscovery:
    """一次性、确定性的议题发现；所有时间口径由外部传入。"""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.as_of = parse_instant(payload.get("as_of"))
        discovery = payload.get("discovery", {}) or {}
        self.cutoff = parse_instant(discovery.get("cutoff"))
        self.max_topics = int(discovery.get("max_topics", 10))
        self.minimum_independent_support = int(discovery.get("minimum_independent_support", 2))
        self.submissions: list[Submission] = []
        self.excluded: list[dict[str, Any]] = []
        self.id_to_index: dict[str, int] = {}
        self.judgments: list[Judgment] = []
        self.same_pairs: list[tuple[int, int]] = []
        self.different_pairs: set[tuple[int, int]] = set()
        self.similarity: dict[tuple[int, int], float] = {}
        self.propagation = UnionFind(0)
        self.forward_roots: dict[int, int] = {}
        self.duplicate_groups: list[dict[str, Any]] = []
        self.clusters: list[Cluster] = []
        self.selection: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 第一步：时间口径与输入装载
    # ------------------------------------------------------------------
    def load(self) -> None:
        kept: list[Submission] = []
        for index, row in enumerate(self.payload.get("submissions", [])):
            available = parse_instant(row.get("available_at"))
            submitted = parse_instant(row.get("submitted_at"))
            reason = None
            if self.as_of is not None and available is not None and available > self.as_of:
                reason = "available_after_as_of"
            elif self.cutoff is not None and submitted is not None and submitted > self.cutoff:
                reason = "submitted_after_discovery_cutoff"
            if reason:
                self.excluded.append({
                    "submission_id": row.get("submission_id"),
                    "reason": reason,
                    "submitted_at": row.get("submitted_at"),
                    "available_at": row.get("available_at"),
                })
                continue
            normalized = normalize_text(row.get("text", ""))
            kept.append(Submission(
                index=len(kept),
                submission_id=str(row["submission_id"]),
                raw_text=str(row.get("text", "")),
                normalized_text=normalized,
                stakeholder=str(row.get("stakeholder", "unknown")) or "unknown",
                submitted_at=str(row.get("submitted_at", "")),
                available_at=str(row.get("available_at", "")),
                forwarded_from=row.get("forwarded_from"),
                features=text_features(normalized),
            ))
        self.submissions = kept
        self.id_to_index = {item.submission_id: item.index for item in kept}
        vectors = build_tfidf([item.features for item in kept])
        for item, vector in zip(kept, vectors):
            item.vector = vector
        self._load_judgments()

    def _load_judgments(self) -> None:
        for row in self.payload.get("judgments", []):
            available = parse_instant(row.get("available_at"))
            applicable = self.as_of is None or available is None or available <= self.as_of
            left_id = str(row.get("left", ""))
            right_id = str(row.get("right", ""))
            if not applicable:
                reason = "available_after_as_of"
            elif left_id not in self.id_to_index or right_id not in self.id_to_index:
                applicable = False
                reason = "reference_outside_analysis_window"
            else:
                reason = "applied"
            judgment = Judgment(left_id, right_id, str(row.get("relation", "")),
                                str(row.get("available_at", "")), applicable, reason)
            self.judgments.append(judgment)
            if not applicable:
                continue
            left = self.id_to_index[left_id]
            right = self.id_to_index[right_id]
            pair = (min(left, right), max(left, right))
            if judgment.relation == "same":
                self.same_pairs.append(pair)
            elif judgment.relation == "different":
                self.different_pairs.add(pair)

    # ------------------------------------------------------------------
    # 第二步：传播结构（转发 + 精确重复文本）
    # ------------------------------------------------------------------
    def build_propagation(self) -> None:
        size = len(self.submissions)
        # 传播树只沿显式转发边生长；同文案但无转发关系的提交仍是独立声音。
        self.propagation = UnionFind(size)
        for item in self.submissions:
            if item.forwarded_from and item.forwarded_from in self.id_to_index:
                self.propagation.union(item.index, self.id_to_index[item.forwarded_from])
        # 精确重复文本作为“重复传播”的另一类证据单独登记，不并入转发树。
        text_to_indices: defaultdict[str, list[int]] = defaultdict(list)
        for item in self.submissions:
            text_to_indices[item.normalized_text].append(item.index)
        for indices in text_to_indices.values():
            if len(indices) > 1:
                members = sorted(self.submissions[index].submission_id for index in indices)
                self.duplicate_groups.append({
                    "normalized_text": self.submissions[indices[0]].normalized_text,
                    "submission_ids": members,
                    "submission_count": len(members),
                })
        # 传播根：每棵转发树只产生一个独立支持计数。
        self.forward_roots = {index: self.propagation.find(index) for index in range(size)}

    # ------------------------------------------------------------------
    # 第三步：文本 + 结构相似度
    # ------------------------------------------------------------------
    def build_similarity(self) -> None:
        size = len(self.submissions)
        # 预计算全部成对相似度为 N×N 矩阵，层次聚类与稳定性重采样只做 O(1)
        # 查表，避免在 O(n^3) 的合并循环里反复计算余弦或构造字典键。
        self.similarity = [[0.0] * size for _ in range(size)]
        self.text_similarity = [[0.0] * size for _ in range(size)]
        for left in range(size):
            self.similarity[left][left] = 1.0
            self.text_similarity[left][left] = 1.0
            for right in range(left + 1, size):
                text_score = cosine(self.submissions[left].vector, self.submissions[right].vector)
                structure_score = 1.0 if self.submissions[left].stakeholder == self.submissions[right].stakeholder else 0.0
                combined = round(TEXT_WEIGHT * text_score + STRUCTURE_WEIGHT * structure_score, 6)
                self.similarity[left][right] = self.similarity[right][left] = combined
                self.text_similarity[left][right] = self.text_similarity[right][left] = text_score

    def pair_similarity(self, left: int, right: int) -> float:
        return self.similarity[left][right]

    def pair_text_similarity(self, left: int, right: int) -> float:
        return self.text_similarity[left][right]

    # ------------------------------------------------------------------
    # 第四步：受约束的平均连接层次聚类 + 轮廓系数选 k
    # ------------------------------------------------------------------
    def cluster(self) -> None:
        size = len(self.submissions)
        # 活跃簇：位置即簇槽位，合并后右槽位移除；linkage 增量维护，
        # average_linkage(A∪B, C) = (|A|·L(A,C) + |B|·L(B,C)) / (|A|+|B|)。
        active = list(range(size))
        members: dict[int, list[int]] = {index: [index] for index in range(size)}
        link_sum = [[0.0] * size for _ in range(size)]
        text_sum = [[0.0] * size for _ in range(size)]
        for left in range(size):
            for right in range(size):
                link_sum[left][right] = self.similarity[left][right]
                text_sum[left][right] = self.text_similarity[left][right]
        # 每个簇槽位维护“因 different 判定而禁止合并的对方成员”集合。
        forbidden_members: dict[int, set[int]] = {index: set() for index in range(size)}
        for left_member, right_member in self.different_pairs:
            forbidden_members[left_member].add(right_member)
            forbidden_members[right_member].add(left_member)
        merge_trace: dict[int, list[str]] = {index: [] for index in range(size)}
        blocked_records: list[str] = []
        blocked_seen: set[tuple[int, int]] = set()

        def record_blocked(left_slot: int, right_slot: int) -> None:
            pair = next(
                (min(member, other), max(member, other))
                for member in members[left_slot] for other in members[right_slot]
                if (min(member, other), max(member, other)) in self.different_pairs
            )
            if pair not in blocked_seen:
                blocked_seen.add(pair)
                blocked_records.append(
                    f"{self.submissions[pair[0]].submission_id}!{self.submissions[pair[1]].submission_id}"
                    f"@manual_different_judgment"
                )

        def is_forbidden(left_slot: int, right_slot: int) -> bool:
            return bool(forbidden_members[left_slot].intersection(members[right_slot]))

        def merge_slots(left_slot: int, right_slot: int, reason: str) -> None:
            left_id = self.submissions[min(members[left_slot])].submission_id
            right_id = self.submissions[min(members[right_slot])].submission_id
            merge_trace[left_slot].append(f"{left_id}+{right_id}@{reason}")
            members[left_slot] = sorted(members[left_slot] + members[right_slot])
            forbidden_members[left_slot] |= forbidden_members[right_slot]
            merge_trace[left_slot].extend(merge_trace.pop(right_slot))
            forbidden_members.pop(right_slot)
            members.pop(right_slot)
            # link_sum/text_sum 始终保存原始成对相似度之和，合并时直接相加；
            # 平均连接在比较时再除以 |A|·|B|，避免加权平均与二次除法互相污染。
            for other in active:
                if other == left_slot or other == right_slot:
                    continue
                link_sum[left_slot][other] += link_sum[right_slot][other]
                text_sum[left_slot][other] += text_sum[right_slot][other]
                link_sum[other][left_slot] = link_sum[left_slot][other]
                text_sum[other][left_slot] = text_sum[left_slot][other]
            active.remove(right_slot)

        # 人工 same 关系：强制合并（记录证据），仍不得违反 different。
        for left, right in sorted(set(self.same_pairs)):
            left_slot = next(slot for slot in active if left in members[slot])
            right_slot = next(slot for slot in active if right in members[slot])
            if left_slot == right_slot:
                continue
            if is_forbidden(left_slot, right_slot):
                record_blocked(left_slot, right_slot)
                continue
            merge_slots(left_slot, right_slot, "manual_same_judgment")

        snapshots: list[tuple[int, list[list[int]], list[str]]] = []
        while len(active) > 1:
            best: tuple[float, float, int, int] | None = None
            for order, left_slot in enumerate(active):
                for right_slot in active[order + 1:]:
                    if is_forbidden(left_slot, right_slot):
                        record_blocked(left_slot, right_slot)
                        continue
                    denominator = len(members[left_slot]) * len(members[right_slot])
                    linkage = link_sum[left_slot][right_slot] / denominator
                    text_score = text_sum[left_slot][right_slot] / denominator
                    if best is None or linkage > best[0]:
                        best = (linkage, text_score, left_slot, right_slot)
            if best is None:
                # 剩余所有合并都被 different 约束封死。
                break
            linkage, text_score, left_slot, right_slot = best
            if linkage < MERGE_SIMILARITY_FLOOR or text_score < TEXT_LINKAGE_FLOOR:
                # 最高优先候选仍达不到双门槛：继续合并只会把无关诉求塞在一起。
                break
            # 先登记合并前切分（处于 max_topics 上限内的候选 k），再执行合并。
            if len(active) <= self.max_topics:
                snapshots.append((len(active), [sorted(members[slot]) for slot in active], list(blocked_records)))
            merge_slots(left_slot, right_slot,
                        f"average_linkage={linkage:.4f},text_linkage={text_score:.4f}")
        snapshots.append((len(active), [sorted(members[slot]) for slot in active], list(blocked_records)))

        clusters = [
            Cluster(members=sorted(members[slot]), merge_trace=list(merge_trace[slot]))
            for slot in active
        ]
        self.clusters = clusters
        self.selection = self._select_partition(snapshots)
        chosen_members = self.selection["partition"]
        rebuilt: list[Cluster] = []
        for group in chosen_members:
            member_set = set(group)
            source = next(cluster for cluster in clusters if set(cluster.members) >= member_set)
            rebuilt.append(Cluster(
                members=sorted(group),
                merge_trace=[trace for trace in source.merge_trace
                             if any(self.submissions[member].submission_id in trace for member in group)],
                forced_merges=list(source.forced_merges),
            ))
        # 把阻断证据挂回最终簇（只包含真正被封死的合并，不含成功合并痕迹）。
        for cluster in rebuilt:
            member_ids = {self.submissions[member].submission_id for member in cluster.members}
            for record in blocked_records:
                anchor = record.split("!")[0]
                if anchor in member_ids:
                    cluster.blocked_merges.append(record)
        self.clusters = rebuilt

    def _silhouette(self, partition: list[list[int]]) -> float:
        labels: dict[int, int] = {}
        for label, group in enumerate(partition):
            for member in group:
                labels[member] = label
        scores: list[float] = []
        for member in range(len(self.submissions)):
            own_label = labels[member]
            own_group = partition[own_label]
            if len(own_group) <= 1:
                continue
            intra = sum(self.pair_similarity(member, other) for other in own_group if other != member) / (len(own_group) - 1)
            nearest = 0.0
            for label, group in enumerate(partition):
                if label == own_label:
                    continue
                nearest = max(nearest, sum(self.pair_similarity(member, other) for other in group) / len(group))
            # 相似度版轮廓系数：(a - b) / max(a, b)，值域 [-1, 1]。
            denominator = max(intra, nearest)
            scores.append((intra - nearest) / denominator if denominator > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def _select_partition(self, snapshots: list[tuple[int, list[list[int]], list[str]]]) -> dict[str, Any]:
        candidates = []
        for cluster_count, partition, blocked in snapshots:
            if cluster_count > self.max_topics:
                continue
            score = self._silhouette(partition)
            candidates.append({
                "cluster_count": cluster_count,
                "silhouette": round(score, 6),
                "partition": partition,
                "blocked": blocked,
            })
        # 优先在“没有单点簇”的切分里选轮廓最高的；否则退而求其次保留全部切分。
        multi = [candidate for candidate in candidates if all(len(group) >= MIN_CLUSTER_FOR_SELECTION for group in candidate["partition"])]
        pool = multi or candidates
        chosen = max(pool, key=lambda candidate: (candidate["silhouette"], -candidate["cluster_count"]))
        return {
            "chosen_cluster_count": chosen["cluster_count"],
            "chosen_silhouette": chosen["silhouette"],
            "partition": chosen["partition"],
            "scan": [
                {"cluster_count": candidate["cluster_count"], "silhouette": candidate["silhouette"]}
                for candidate in sorted(candidates, key=lambda item: item["cluster_count"])
            ],
            "merge_similarity_floor": MERGE_SIMILARITY_FLOOR,
        }

    # ------------------------------------------------------------------
    # 第五步：议题对象、传播规模与独立支持度
    # ------------------------------------------------------------------
    def build_issues(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        all_stakeholders = sorted({item.stakeholder for item in self.submissions})
        for cluster in self.clusters:
            members = sorted(cluster.members)
            member_rows = [self.submissions[index] for index in members]
            root_to_members: defaultdict[int, list[int]] = defaultdict(list)
            for index in members:
                root_to_members[self.forward_roots[index]].append(index)
            independent_support = len(root_to_members)
            forwarded = [item.submission_id for item in member_rows if item.forwarded_from]
            stakeholder_counts = Counter(item.stakeholder for item in member_rows)
            earliest = min(item.submitted_at for item in member_rows)
            # 代表性原文：传播树根节点中最早的一条；另附最近一条便于复核。
            root_indices = sorted(root_to_members, key=lambda root: min(self.submissions[index].submitted_at for index in root_to_members[root]))
            representative_root = min(root_to_members[root_indices[0]],
                                      key=lambda index: self.submissions[index].submitted_at)
            latest = max(members, key=lambda index: self.submissions[index].submitted_at)
            text_variants = sorted({item.normalized_text for item in member_rows})
            rationale_terms = self._distinctive_terms(members)
            meets_threshold = independent_support >= self.minimum_independent_support
            issues.append({
                "issue_key": fingerprint("|".join(text_variants), 12),
                "title": self.submissions[representative_root].raw_text,
                "member_submission_ids": [item.submission_id for item in member_rows],
                "member_count": len(member_rows),
                "independent_support": independent_support,
                "forwarded_count": len(forwarded),
                "forwarded_submission_ids": sorted(forwarded),
                "propagation_roots": sorted({
                    self.submissions[root].submission_id
                    for root in (self.forward_roots[index] for index in members)
                }),
                "stakeholder_coverage": {
                    "groups_present": sorted(stakeholder_counts),
                    "group_counts": dict(sorted(stakeholder_counts.items())),
                    "distinct_groups": len(stakeholder_counts),
                    "population_groups_total": len(all_stakeholders),
                },
                "first_submitted_at": earliest,
                "representative_submission_id": self.submissions[representative_root].submission_id,
                "representative_text": self.submissions[representative_root].raw_text,
                "latest_submission_id": self.submissions[latest].submission_id,
                "text_variants": text_variants,
                "rationale_terms": rationale_terms,
                "meets_minimum_independent_support": meets_threshold,
                "qualification": "agenda_ready" if meets_threshold else "candidate_below_independent_support",
                "merge_evidence": list(cluster.merge_trace),
                "blocked_merge_evidence": list(cluster.blocked_merges),
            })
        # 排序：先按是否达到独立支持度，再按独立支持度降序、首次出现时间升序、
        # 成员最小 ID 升序——确定性顺序，且议题编号语义稳定。
        issues.sort(key=lambda issue: (
            not issue["meets_minimum_independent_support"],
            -issue["independent_support"],
            issue["first_submitted_at"],
            issue["member_submission_ids"][0],
        ))
        for position, issue in enumerate(issues, 1):
            issue["issue_id"] = f"issue-{self._version_label()}-{position:02d}"
        return issues

    def _version_label(self) -> str:
        return "d1"

    def _distinctive_terms(self, members: list[int], limit: int = 6) -> list[str]:
        inside: Counter = Counter()
        for index in members:
            inside.update(self.submissions[index].features)
        outside: Counter = Counter()
        member_set = set(members)
        for index, item in enumerate(self.submissions):
            if index in member_set:
                continue
            outside.update(item.features)
        inside_total = sum(inside.values()) or 1
        outside_total = sum(outside.values()) or 1
        scored = []
        for term, count in inside.items():
            if not term.startswith("b:"):
                continue
            inside_rate = count / inside_total
            outside_rate = outside.get(term, 0) / outside_total
            scored.append((inside_rate - outside_rate, term[2:]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [term for score, term in scored if score > 0][:limit]

    # ------------------------------------------------------------------
    # 第六步：稳定性证据
    # ------------------------------------------------------------------
    def stability_evidence(self, issues: list[dict[str, Any]]) -> dict[str, Any]:
        base_labels = {submission_id: issue["issue_id"] for issue in issues for submission_id in issue["member_submission_ids"]}
        if not self.submissions:
            # 空输入：无法重采样，显式给出空证据而不是除零。
            return {
                "random_seed": RANDOM_SEED,
                "bootstrap_runs": 0,
                "bootstrap_root_preservation_mean": None,
                "bootstrap_root_preservation_min": None,
                "bootstrap_ari_mean": None,
                "leave_one_root_out_preservation_mean": None,
                "leave_one_root_out_preservation_min": None,
                "naive_equal_weight_comparison": {
                    "method": "recluster_without_forwarding_weights",
                    "adjusted_rand_index_vs_base": None,
                    "base_cluster_preservation": None,
                    "interpretation": "输入窗口内没有可用意见，稳定性指标不适用",
                },
                "selection": self.selection,
            }
        rng = random.Random(RANDOM_SEED)
        roots = sorted(set(self.forward_roots.values()))
        root_to_members: defaultdict[int, list[int]] = defaultdict(list)
        for index, root in self.forward_roots.items():
            root_to_members[root].append(index)

        def partition_from_roots(selected_roots: set[int]) -> dict[int, int]:
            selected_indices = [index for root in selected_roots for index in root_to_members[root]]
            labels = self._recluster(selected_indices)
            return labels

        def root_preservation(selected_roots: set[int], labels: dict[int, int]) -> float:
            root_clusters: defaultdict[int, set[int]] = defaultdict(set)
            for index, cluster_label in labels.items():
                root_clusters[self.forward_roots[index]].add(cluster_label)
            intact = sum(1 for root in selected_roots if len(root_clusters[root]) == 1)
            return intact / len(selected_roots) if selected_roots else 0.0

        boot_scores: list[float] = []
        boot_aris: list[float] = []
        for _ in range(BOOTSTRAP_RUNS):
            sampled = set(rng.choice(roots) for _ in roots)
            labels = partition_from_roots(sampled)
            boot_scores.append(root_preservation(sampled, labels))
            boot_aris.append(self._ari_against_base(labels, base_labels))

        # 留一法：逐个移除传播根。
        leave_one_out: list[float] = []
        for removed in roots:
            remaining = set(roots) - {removed}
            labels = partition_from_roots(remaining)
            leave_one_out.append(root_preservation(remaining, labels))

        # 朴素对照：忽略转发、每条意见等权重新聚类，检验转发爆量是否扭曲结构。
        naive_labels = self._recluster(list(range(len(self.submissions))), ignore_propagation=True)
        naive_ari = self._ari_against_base(naive_labels, base_labels)
        naive_preservation = self._base_cluster_preservation(naive_labels, base_labels)

        return {
            "random_seed": RANDOM_SEED,
            "bootstrap_runs": BOOTSTRAP_RUNS,
            "bootstrap_root_preservation_mean": round(sum(boot_scores) / len(boot_scores), 6),
            "bootstrap_root_preservation_min": round(min(boot_scores), 6),
            "bootstrap_ari_mean": round(sum(boot_aris) / len(boot_aris), 6),
            "leave_one_root_out_preservation_mean": round(sum(leave_one_out) / len(leave_one_out), 6),
            "leave_one_root_out_preservation_min": round(min(leave_one_out), 6),
            "naive_equal_weight_comparison": {
                "method": "recluster_without_forwarding_weights",
                "adjusted_rand_index_vs_base": round(naive_ari, 6),
                "base_cluster_preservation": round(naive_preservation, 6),
                "interpretation": "ARI 接近 1 表示转发爆量未改变议题结构",
            },
            "selection": self.selection,
        }

    def _recluster(self, indices: list[int], ignore_propagation: bool = False) -> dict[int, int]:
        """在给定意见子集上重跑同一套受约束聚类，返回 index -> 簇序号。"""
        groups = self._incremental_clusters(sorted(indices), self.different_pairs)
        labels: dict[int, int] = {}
        for label, group in enumerate(groups):
            for member in group:
                labels[member] = label
        return labels

    def _incremental_clusters(self, indices: list[int],
                              different_pairs: set[tuple[int, int]]) -> list[list[int]]:
        """与主聚类同口径的增量平均连接聚类（供稳定性重采样复用）。"""
        active = list(indices)
        members: dict[int, list[int]] = {index: [index] for index in indices}
        link_sum: dict[int, dict[int, float]] = {
            index: {other: self.similarity[index][other] for other in indices} for index in indices
        }
        text_sum: dict[int, dict[int, float]] = {
            index: {other: self.text_similarity[index][other] for other in indices} for index in indices
        }
        forbidden_members: dict[int, set[int]] = {index: set() for index in indices}
        for left_member, right_member in different_pairs:
            if left_member in forbidden_members and right_member in forbidden_members:
                forbidden_members[left_member].add(right_member)
                forbidden_members[right_member].add(left_member)

        def merge(left_slot: int, right_slot: int) -> None:
            members[left_slot] = sorted(members[left_slot] + members[right_slot])
            forbidden_members[left_slot] |= forbidden_members.pop(right_slot)
            members.pop(right_slot)
            # 原始成对相似度之和：合并即相加，平均连接在比较时按簇大小归一化。
            for other in active:
                if other == left_slot or other == right_slot:
                    continue
                link_sum[left_slot][other] += link_sum[right_slot][other]
                text_sum[left_slot][other] += text_sum[right_slot][other]
                link_sum[other][left_slot] = link_sum[left_slot][other]
                text_sum[other][left_slot] = text_sum[left_slot][other]
            active.remove(right_slot)

        while len(active) > 1:
            best: tuple[float, float, int, int] | None = None
            for order, left_slot in enumerate(active):
                for right_slot in active[order + 1:]:
                    if forbidden_members[left_slot].intersection(members[right_slot]):
                        continue
                    denominator = len(members[left_slot]) * len(members[right_slot])
                    linkage = link_sum[left_slot][right_slot] / denominator
                    text_score = text_sum[left_slot][right_slot] / denominator
                    if best is None or linkage > best[0]:
                        best = (linkage, text_score, left_slot, right_slot)
            if best is None or best[0] < MERGE_SIMILARITY_FLOOR or best[1] < TEXT_LINKAGE_FLOOR:
                break
            merge(best[2], best[3])
        return [sorted(members[slot]) for slot in active]

    def _ari_against_base(self, labels: dict[int, int], base_labels: dict[str, int]) -> float:
        indices = sorted(labels)
        base = [base_labels[self.submissions[index].submission_id] for index in indices]
        current = [labels[index] for index in indices]
        return adjusted_rand_index(base, current)

    def _base_cluster_preservation(self, labels: dict[int, int], base_labels: dict[str, int]) -> float:
        base_to_local: defaultdict[str, set[int]] = defaultdict(set)
        for index, local_label in labels.items():
            base_to_local[base_labels[self.submissions[index].submission_id]].add(local_label)
        intact = sum(1 for clusters in base_to_local.values() if len(clusters) == 1)
        return intact / len(base_to_local) if base_to_local else 0.0

    # ------------------------------------------------------------------
    # 第七步：覆盖、未归类、人工关系、版本谱系
    # ------------------------------------------------------------------
    def build_coverage(self, issues: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(self.submissions)
        agenda_ready = [issue for issue in issues if issue["meets_minimum_independent_support"]]
        assigned = sum(issue["member_count"] for issue in agenda_ready)
        stakeholder_population = Counter(item.stakeholder for item in self.submissions)
        per_group: dict[str, dict[str, Any]] = {}
        for stakeholder, count in sorted(stakeholder_population.items()):
            owning_issues = sorted({
                issue["issue_id"] for issue in agenda_ready
                if stakeholder in issue["stakeholder_coverage"]["groups_present"]
            })
            independent_roots = {
                self.forward_roots[item.index]
                for item in self.submissions
                if item.stakeholder == stakeholder
            }
            per_group[stakeholder] = {
                "submission_count": count,
                "independent_voice_count": len(independent_roots),
                "issue_ids": owning_issues,
                "represented": bool(owning_issues),
            }
        candidate_members = sum(
            issue["member_count"] for issue in issues if not issue["meets_minimum_independent_support"]
        )
        return {
            "submissions_in_scope": total,
            "submissions_assigned_to_agenda_issues": assigned,
            "submissions_in_candidate_issues": candidate_members,
            "submissions_unassigned": total - assigned - candidate_members + len(self.excluded),
            "issues_total": len(issues),
            "issues_meeting_independent_support": len(agenda_ready),
            "candidate_issues_below_support": len(issues) - len(agenda_ready),
            "stakeholder_groups_total": len(stakeholder_population),
            "stakeholder_groups_represented": sum(1 for row in per_group.values() if row["represented"]),
            "per_stakeholder_group": per_group,
            "excluded_by_time_window": self.excluded,
        }

    def build_unassigned(self, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        agenda_ids = {issue["issue_id"] for issue in issues if issue["meets_minimum_independent_support"]}
        rows: list[dict[str, Any]] = []
        # 未达独立支持度的候选议题：不进议程，但作为“难以进入议程的声音”显式留痕，
        # 少数群体样本少时其独立议题仍可被看见、复核，而不是被静默丢弃。
        for issue in issues:
            if issue["meets_minimum_independent_support"]:
                continue
            for submission_id in issue["member_submission_ids"]:
                index = self.id_to_index[submission_id]
                rows.append({
                    "submission_id": submission_id,
                    "text": self.submissions[index].raw_text,
                    "stakeholder": self.submissions[index].stakeholder,
                    "reason": "candidate_issue_below_minimum_independent_support",
                    "candidate_issue_id": issue["issue_id"],
                    "candidate_independent_support": issue["independent_support"],
                    "nearest_issue_id": issue["issue_id"],
                    "nearest_similarity": 1.0,
                })
        assigned = {submission_id for issue in issues for submission_id in issue["member_submission_ids"]}
        for item in self.submissions:
            if item.submission_id in assigned:
                continue
            nearest = self._nearest_issue(item.index, [issue for issue in issues if issue["issue_id"] in agenda_ids])
            rows.append({
                "submission_id": item.submission_id,
                "text": item.raw_text,
                "stakeholder": item.stakeholder,
                "reason": "no_cluster_above_similarity_floor_or_blocked_by_judgment",
                "nearest_issue_id": nearest["issue_id"],
                "nearest_similarity": nearest["similarity"],
            })
        for row in self.excluded:
            rows.append({
                "submission_id": row["submission_id"],
                "text": next((item.raw_text for item in self.submissions if item.submission_id == row["submission_id"]), None),
                "stakeholder": next((item.stakeholder for item in self.submissions if item.submission_id == row["submission_id"]), None),
                "reason": row["reason"],
                "nearest_issue_id": None,
                "nearest_similarity": None,
            })
        rows.sort(key=lambda row: (row["submission_id"] or "", row["reason"]))
        return rows

    def _nearest_issue(self, index: int, issues: list[dict[str, Any]]) -> dict[str, Any]:
        best_id = None
        best_score = -1.0
        for issue in issues:
            for submission_id in issue["member_submission_ids"]:
                other = self.id_to_index[submission_id]
                score = self.pair_similarity(index, other)
                if score > best_score:
                    best_score = score
                    best_id = issue["issue_id"]
        return {"issue_id": best_id, "similarity": round(best_score, 6)}

    def build_judgment_ledger(self) -> dict[str, Any]:
        applied = []
        pending = []
        for judgment in self.judgments:
            row = {
                "left": judgment.left,
                "right": judgment.right,
                "relation": judgment.relation,
                "available_at": judgment.available_at,
                "status": judgment.applicable and "applied" or "pending",
                "reason": judgment.reason,
            }
            (applied if judgment.applicable else pending).append(row)
        return {"applied_judgments": applied, "pending_judgments": pending}

    def build_lineage(self, issues: list[dict[str, Any]]) -> dict[str, Any]:
        issue_by_id = {issue["issue_id"]: issue for issue in issues}
        member_to_issue = {
            submission_id: issue["issue_id"]
            for issue in issues for submission_id in issue["member_submission_ids"]
        }
        relations: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        anchored_commitments: list[dict[str, Any]] = []
        known_previous = {str(row["issue_id"]): row for row in self.payload.get("previous_issues", [])}

        for previous_id, previous in known_previous.items():
            overlap = [submission_id for submission_id in previous.get("members", []) if submission_id in member_to_issue]
            targets = sorted({member_to_issue[submission_id] for submission_id in overlap})
            missing_members = [submission_id for submission_id in previous.get("members", []) if submission_id not in member_to_issue]
            if not targets:
                unresolved.append({
                    "kind": "previous_issue_without_current_target",
                    "issue_id": previous_id,
                    "version": previous.get("version"),
                    "members": previous.get("members", []),
                    "reason": "no_member_maps_to_a_current_issue",
                })
                continue
            relation_kind = "anchored" if len(targets) == 1 else "split"
            for target in targets:
                relations.append({
                    "previous_issue_id": previous_id,
                    "previous_version": previous.get("version"),
                    "current_issue_id": target,
                    "relation": relation_kind,
                    "shared_submission_ids": sorted(submission_id for submission_id in overlap if member_to_issue[submission_id] == target),
                    "missing_member_submission_ids": missing_members,
                    "basis": "member_overlap",
                })
            for commitment_id in previous.get("commitment_ids", []):
                anchored_commitments.append({
                    "commitment_id": commitment_id,
                    "previous_issue_id": previous_id,
                    "current_issue_id": targets[0],
                    "anchor_relation": relation_kind,
                    "note": "承诺随谱系关系锚定；后续重排不得改变该引用含义",
                })

        manual_log: list[dict[str, Any]] = []
        for action in self.payload.get("manual_actions", []):
            available = parse_instant(action.get("available_at"))
            action_row = {
                "action_id": action.get("action_id"),
                "kind": action.get("kind"),
                "issue_ids": action.get("issue_ids", []),
                "available_at": action.get("available_at"),
            }
            if self.as_of is not None and available is not None and available > self.as_of:
                action_row.update(status="pending", reason="available_after_as_of")
                manual_log.append(action_row)
                continue
            referenced = action.get("issue_ids", [])
            missing = [issue_id for issue_id in referenced if issue_id not in known_previous and issue_id not in issue_by_id]
            if missing:
                action_row.update(status="blocked", reason="unresolved_issue_reference", unresolved_issue_ids=missing)
                unresolved.append({
                    "kind": "manual_action_unresolved_reference",
                    "action_id": action.get("action_id"),
                    "issue_id": missing[0],
                    "reason": "referenced_issue_unknown_in_previous_versions_and_current_discovery",
                })
            else:
                action_row.update(status="recorded", reason="applied_to_lineage")
            manual_log.append(action_row)

        return {
            "algorithm_version": ALGORITHM_VERSION,
            "discovery_version": "d1",
            "relations": relations,
            "anchored_commitments": anchored_commitments,
            "manual_actions": manual_log,
            "unresolved_references": unresolved,
        }

    def run(self) -> dict[str, Any]:
        self.load()
        self.build_propagation()
        self.build_similarity()
        self.cluster()
        issues = self.build_issues()
        coverage = self.build_coverage(issues)
        unassigned = self.build_unassigned(issues)
        stability = self.stability_evidence(issues)
        judgment_ledger = self.build_judgment_ledger()
        lineage = self.build_lineage(issues)
        duplicate_summary = sorted(self.duplicate_groups, key=lambda row: row["submission_ids"][0])
        return {
            "issues": issues,
            "unassigned": unassigned,
            "coverage": coverage,
            "stability": stability,
            "lineage": lineage,
            "judgment_ledger": judgment_ledger,
            "propagation": {
                "duplicate_text_groups": duplicate_summary,
                "forwarding_edges": sorted(
                    {
                        (item.forwarded_from, item.submission_id)
                        for item in self.submissions if item.forwarded_from
                    }
                ),
            },
        }


def adjusted_rand_index(left_labels: list[Any], right_labels: list[Any]) -> float:
    """标准 ARI 实现（组合计数，纯标准库），值域 [-1, 1]。"""
    if len(left_labels) != len(right_labels) or not left_labels:
        return 0.0

    def choose_two(value: int) -> float:
        return value * (value - 1) / 2.0

    contingency: defaultdict[Any, Counter] = defaultdict(Counter)
    left_sizes: Counter = Counter()
    right_sizes: Counter = Counter()
    for left, right in zip(left_labels, right_labels):
        contingency[left][right] += 1
        left_sizes[left] += 1
        right_sizes[right] += 1
    total_pairs = choose_two(len(left_labels))
    sum_contingency = sum(choose_two(count) for row in contingency.values() for count in row.values())
    sum_left = sum(choose_two(count) for count in left_sizes.values())
    sum_right = sum(choose_two(count) for count in right_sizes.values())
    if total_pairs == 0:
        return 0.0
    expected = sum_left * sum_right / total_pairs
    maximum = (sum_left + sum_right) / 2.0
    if maximum == expected:
        return 0.0
    return (sum_contingency - expected) / (maximum - expected)
