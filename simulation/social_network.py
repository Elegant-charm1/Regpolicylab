"""
社交网络传播模拟

升级版政策传播网络：
- 属性驱动的加权有向图
- 带固执度的 FJ 信念传播
- 个性化论坛曝光与互动
- 轻量级动态边权更新
- 每步影响解释轨迹
"""

from __future__ import annotations

import math
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .models import PropagationStep
from .policy_config import get_policy_simulation_config
from util.PolicyForumDB import (
    FORUM_DB_PATH,
    create_post,
    get_forum_statistics,
    get_top_posts,
    init_policy_forum,
    react_to_post,
)


class _DefaultForumService:
    """兼容未显式注入 ForumService 的旧用法。"""

    def __init__(self) -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(project_root, FORUM_DB_PATH)
        init_policy_forum(self.db_path)

    def reset(self) -> str:
        return init_policy_forum(self.db_path)

    def create_post(
        self,
        agent_id: str,
        agent_role: str,
        agent_type: str,
        content: str,
        stance: str,
        step: int,
    ) -> int:
        return create_post(
            agent_id, agent_role, agent_type, content, stance, step, self.db_path
        )

    def react_to_post(self, agent_id: str, post_id: int, reaction_type: str) -> bool:
        return react_to_post(agent_id, post_id, reaction_type, self.db_path)

    def get_top_posts(self, step: int, limit: int = 10) -> List[Dict[str, Any]]:
        return get_top_posts(step, limit, self.db_path)

    def get_forum_statistics(self, step: int) -> Dict[str, Any]:
        return get_forum_statistics(step, self.db_path)


class SocialNetwork:
    """社交网络传播模拟类"""

    STANCE_TO_BELIEF = {"保守": -0.5, "中立": 0.0, "激进": 0.5}
    ROLE_LAYOUT_ORDER = [
        "regulator",
        "manager",
        "auditor",
        "accountant",
        "investor",
    ]
    ROLE_CLUSTER_POSITIONS = {
        "regulator": {"x": 0.50, "y": 0.18},
        "manager": {"x": 0.79, "y": 0.36},
        "investor": {"x": 0.73, "y": 0.77},
        "accountant": {"x": 0.27, "y": 0.77},
        "auditor": {"x": 0.21, "y": 0.36},
    }

    def __init__(
        self,
        agents: List[Dict[str, Any]],
        forum_service: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or get_policy_simulation_config()
        social_config = self.config.get("social_network", {})

        self.DEFAULT_TOP_K = int(social_config.get("default_top_k", 5))
        self.EXPOSURE_LIMIT = int(social_config.get("exposure_limit", 5))
        self.CONFIDENCE_THRESHOLD = float(
            social_config.get("confidence_threshold", 0.7)
        )
        self.CONFIDENCE_DECAY = float(social_config.get("confidence_decay", 0.2))
        self.REINFORCE_DELTA = float(social_config.get("reinforce_delta", 0.05))
        self.WEAKEN_DELTA = float(social_config.get("weaken_delta", 0.05))
        self.MIN_EDGE_WEIGHT = float(social_config.get("min_edge_weight", 0.05))
        self.WEIGHT_CONFIG = social_config.get("weight_config", {})
        self.ROLE_AFFINITY = social_config.get("role_affinity", {})
        self.ROLE_AUTHORITY = social_config.get("role_authority", {})

        self.agents = agents
        self.graph: Dict[str, Dict[str, float]] = {}  # agent_id -> {agent_id -> weight}
        self.forum_posts: Dict[str, int] = {}
        self.step_posts: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.agent_by_id: Dict[str, Dict[str, Any]] = {}  # agent_id -> agent dict
        self.latest_step_metrics = {
            "active_posters_count": 0,
            "forum_exposed_count": 0,
        }
        self.forum_service = forum_service or _DefaultForumService()

        self.build_network()

    def build_network(self):
        """构建属性驱动的加权有向图。"""
        self.graph = {}
        self.agent_by_id = {agent["id"]: agent for agent in self.agents}
        self._initialize_agent_states()

        if len(self.agents) <= 1:
            self.graph = {self.agents[0]["id"]: {}} if self.agents else {}
            return

        top_k = min(self.DEFAULT_TOP_K, len(self.agents) - 1)
        for source_agent in self.agents:
            source_id = source_agent["id"]
            candidates = []
            for target_agent in self.agents:
                if source_agent["id"] == target_agent["id"]:
                    continue
                weight = self._compute_edge_weight(source_agent, target_agent)
                if weight > 0:
                    candidates.append((target_agent["id"], weight))

            candidates.sort(key=lambda item: item[1], reverse=True)
            selected = candidates[:top_k]
            total_weight = sum(weight for _, weight in selected)
            if total_weight <= 0:
                self.graph[source_id] = {}
                continue

            self.graph[source_id] = {
                target_id: weight / total_weight
                for target_id, weight in selected
            }

    def simulate_propagation(self, agents: List[Dict[str, Any]], steps: int = 10):
        """模拟政策解读传播 + 论坛互动。"""
        self.agents = agents
        self.forum_posts = {}
        self.step_posts = defaultdict(list)
        self.agent_by_id = {agent["id"]: agent for agent in self.agents}

        if not agents:
            empty_network = {
                "nodes": [],
                "edges": [],
                "meta": {
                    "step": 0,
                    "total_steps": steps,
                    "node_count": 0,
                    "edge_count": 0,
                    "role_counts": {},
                },
            }
            return {
                "final_beliefs": {},
                "history": [],
                "network": empty_network,
                "statistics": {},
                "polarization": 0.0,
                "forum_posts": [],
            }

        beliefs = {}
        for agent in agents:
            agent_id = agent["id"]
            belief = self._resolve_initial_belief(agent)
            agent["belief"] = belief
            agent["initial_belief"] = belief
            beliefs[agent_id] = belief

        self.build_network()
        propagation_history = []
        final_network_snapshot = self._get_network_data(
            beliefs,
            step=0,
            total_steps=steps,
            statistics=self._calculate_statistics(beliefs),
        )

        for step in range(steps):
            step_posts = self._publish_posts(step, beliefs)
            top_posts = self.forum_service.get_top_posts(step, limit=20)

            forum_effects = {}
            interaction_signals = defaultdict(lambda: defaultdict(float))
            exposed_count = 0
            forum_traces: Dict[str, Dict[str, Any]] = {}

            for agent in agents:
                agent_id = agent["id"]
                exposure_pool = self._get_exposure_pool(agent_id, step_posts, top_posts)
                if exposure_pool:
                    exposed_count += 1
                (
                    forum_effect,
                    agent_signals,
                    adopted_count,
                    forum_trace,
                ) = self._calculate_forum_influence(agent_id, agent, exposure_pool, beliefs)
                forum_effects[agent_id] = forum_effect
                forum_traces[agent_id] = forum_trace
                forum_traces[agent_id]["adopted_count"] = adopted_count
                for author_id, signal_value in agent_signals.items():
                    interaction_signals[agent_id][author_id] += signal_value

            new_beliefs = {}
            influence_trace = {}
            for agent in agents:
                agent_id = agent["id"]
                previous_belief = beliefs[agent_id]
                initial_belief = float(agent.get("initial_belief", previous_belief))
                stubbornness = float(agent.get("stubbornness", 0.5))
                forum_weight = self._calculate_forum_weight(agent)
                social_influence = self._calculate_social_influence(agent_id, beliefs)
                forum_influence = forum_effects.get(agent_id, previous_belief)

                social_component = (1 - forum_weight) * social_influence
                forum_component = forum_weight * forum_influence
                base_component = stubbornness * initial_belief
                updated_belief = base_component + (1 - stubbornness) * (
                    social_component + forum_component
                )
                updated_belief = self._clamp(updated_belief, -1.0, 1.0)
                new_beliefs[agent_id] = round(updated_belief, 4)

                trace = forum_traces.get(
                    agent_id, {"exposed_post_ids": [], "adopted_post_ids": []}
                )
                influence_trace[agent_id] = {
                    "previous_belief": round(previous_belief, 4),
                    "initial_belief": round(initial_belief, 4),
                    "stubbornness": round(stubbornness, 4),
                    "forum_weight": round(forum_weight, 4),
                    "social_influence": round(social_influence, 4),
                    "forum_influence": round(forum_influence, 4),
                    "updated_belief": round(updated_belief, 4),
                    "initial_component": round(base_component, 4),
                    "social_component": round((1 - stubbornness) * social_component, 4),
                    "forum_component": round((1 - stubbornness) * forum_component, 4),
                    "exposed_post_ids": trace.get("exposed_post_ids", []),
                    "adopted_post_ids": trace.get("adopted_post_ids", []),
                    "adopted_count": trace.get("adopted_count", 0),
                    "social_neighbors": self._get_social_neighbor_trace(agent_id),
                }

            beliefs = new_beliefs
            for agent in agents:
                agent["belief"] = beliefs[agent["id"]]

            if (step + 1) % 2 == 0:
                self._update_edge_weights(interaction_signals, beliefs)

            self.latest_step_metrics = {
                "active_posters_count": len(step_posts),
                "forum_exposed_count": exposed_count,
            }
            forum_stats = self._augment_forum_stats(
                self.forum_service.get_forum_statistics(step)
            )
            step_statistics = self._calculate_statistics(beliefs)
            network_snapshot = self._get_network_data(
                beliefs,
                step=step,
                total_steps=steps,
                statistics=step_statistics,
            )

            step_result = PropagationStep(
                step=step,
                beliefs={k: round(v, 3) for k, v in beliefs.items()},
                network_snapshot=network_snapshot,
                forum_stats=forum_stats,
                statistics=step_statistics,
                influence_trace=influence_trace,
            )
            propagation_history.append(step_result.to_dict())
            final_network_snapshot = network_snapshot

        final_statistics = self._calculate_statistics(beliefs)
        final_network_snapshot = self._get_network_data(
            beliefs,
            step=max(steps - 1, 0),
            total_steps=steps,
            statistics=final_statistics,
        )

        return {
            "final_beliefs": {k: round(v, 3) for k, v in beliefs.items()},
            "history": propagation_history,
            "network": final_network_snapshot,
            "statistics": final_statistics,
            "polarization": final_statistics.get("polarization_index", 0.0),
            "forum_posts": self.forum_service.get_top_posts(max(steps - 1, 0), limit=20),
        }

    def _initialize_agent_states(self):
        for agent in self.agents:
            current_belief = float(agent.get("belief", 0.0))
            agent["belief"] = self._clamp(current_belief, -1.0, 1.0)
            agent.setdefault("initial_belief", agent["belief"])
            agent.setdefault("stubbornness", 0.5)
            agent.setdefault("activity_level", 0.55)
            agent.setdefault("susceptibility", 0.5)
            agent.setdefault("expression_threshold", 0.18)
            agent.setdefault("repost_tendency", 0.2)

            characteristics = agent.setdefault("characteristics", {})
            if "peer_influence" not in characteristics:
                characteristics["peer_influence"] = round(
                    0.3 + (1 - float(agent.get("stubbornness", 0.5))) * 0.5,
                    2,
                )

    def _compute_edge_weight(self, source_agent, target_agent):
        role_affinity = self.ROLE_AFFINITY.get(source_agent["role"], {}).get(
            target_agent["role"], 0.4
        )
        profile_similarity = self._compute_profile_similarity(
            source_agent.get("profile"), target_agent.get("profile")
        )
        concern_similarity = self._compute_concern_similarity(
            source_agent.get("concerns", []), target_agent.get("concerns", [])
        )
        belief_similarity = self._compute_belief_similarity(
            float(source_agent.get("initial_belief", source_agent.get("belief", 0.0))),
            float(target_agent.get("initial_belief", target_agent.get("belief", 0.0))),
        )
        authority_score = self._get_authority_score(target_agent)

        weight = (
            self.WEIGHT_CONFIG.get("role_affinity", 0.35) * role_affinity
            + self.WEIGHT_CONFIG.get("profile_similarity", 0.20) * profile_similarity
            + self.WEIGHT_CONFIG.get("concern_similarity", 0.15) * concern_similarity
            + self.WEIGHT_CONFIG.get("belief_similarity", 0.15) * belief_similarity
            + self.WEIGHT_CONFIG.get("authority", 0.15) * authority_score
        )
        return max(weight, 0.0)

    def _compute_profile_similarity(self, source_profile, target_profile):
        source_tokens = self._extract_profile_tokens(source_profile)
        target_tokens = self._extract_profile_tokens(target_profile)
        if not source_tokens or not target_tokens:
            return 0.0

        intersection = source_tokens & target_tokens
        union = source_tokens | target_tokens
        return len(intersection) / len(union) if union else 0.0

    def _extract_profile_tokens(self, profile):
        if not isinstance(profile, dict):
            return set()

        keys = [
            "organization_type",
            "specialization",
            "industry_focus",
            "education_background",
            "strategy",
            "followed_industries",
            "self_description",
        ]
        tokens = set()
        for key in keys:
            value = profile.get(key)
            if not value:
                continue
            items = value if isinstance(value, list) else [value]
            for item in items:
                text = str(item).strip().lower()
                if not text:
                    continue
                for token in re.split(r"[\s,，、/]+", text):
                    if token:
                        tokens.add(token)
        return tokens

    def _compute_concern_similarity(self, source_concerns, target_concerns):
        source_set = set(source_concerns or [])
        target_set = set(target_concerns or [])
        if not source_set or not target_set:
            return 0.0

        intersection = source_set & target_set
        union = source_set | target_set
        return len(intersection) / len(union) if union else 0.0

    def _compute_belief_similarity(self, source_belief, target_belief):
        difference = abs(source_belief - target_belief)
        return max(0.0, 1.0 - difference / 2.0)

    def _get_authority_score(self, agent):
        role_bonus = self.ROLE_AUTHORITY.get(agent.get("role"), 0.5)
        profile = agent.get("profile", {}) or {}
        network_influence = float(profile.get("network_influence", 0.5))
        reputation_score = float(profile.get("reputation_score", 0.5))
        influence_tier_bonus = {
            "high": 0.12,
            "medium": 0.04,
            "low": -0.03,
        }.get(agent.get("influence_tier"), 0.0)
        authority = (
            0.2 * role_bonus
            + 0.45 * network_influence
            + 0.35 * reputation_score
            + influence_tier_bonus
        )
        return self._clamp(authority, 0.0, 1.0)

    def _resolve_initial_belief(self, agent):
        if agent.get("reaction"):
            stance = agent["reaction"].get("decision_stance", "中立")
            fallback = self.STANCE_TO_BELIEF.get(stance, 0.0)
            belief_score = agent["reaction"].get("belief_score", fallback)
            try:
                return self._clamp(float(belief_score), -1.0, 1.0)
            except (TypeError, ValueError):
                return fallback

        try:
            return self._clamp(float(agent.get("belief", 0.0)), -1.0, 1.0)
        except (TypeError, ValueError):
            return 0.0

    def _publish_posts(self, step, beliefs):
        posts = []
        for agent in self.agents:
            agent_id = agent["id"]
            if not agent.get("reaction"):
                continue
            if not self._should_publish(agent, beliefs[agent_id]):
                continue

            content = self._compose_post_content(agent)
            if not content:
                continue

            stance = self._belief_to_stance(beliefs[agent_id])
            post_id = self.forum_service.create_post(
                agent_id,
                agent["role"],
                agent.get("agent_type", "unknown"),
                content,
                stance,
                step,
            )
            post_data = {
                "id": post_id,
                "agent_id": agent_id,
                "agent_role": agent["role"],
                "agent_type": agent.get("agent_type", "unknown"),
                "content": content,
                "stance": stance,
                "score": 0,
                "likes": 0,
                "unlikes": 0,
                "author_id": agent_id,
                "belief": beliefs[agent_id],
                "author_influence": self._get_authority_score(agent),
            }
            posts.append(post_data)
            self.forum_posts[agent_id] = post_id

        if not posts and self.agents:
            fallback_agent = max(
                self.agents,
                key=lambda agent: agent.get("activity_level", 0.0),
            )
            fallback_id = fallback_agent["id"]
            if fallback_agent.get("reaction"):
                content = self._compose_post_content(fallback_agent)
                if content:
                    stance = self._belief_to_stance(beliefs[fallback_id])
                    post_id = self.forum_service.create_post(
                        fallback_id,
                        fallback_agent["role"],
                        fallback_agent.get("agent_type", "unknown"),
                        content,
                        stance,
                        step,
                    )
                    posts.append(
                        {
                            "id": post_id,
                            "agent_id": fallback_id,
                            "agent_role": fallback_agent["role"],
                            "agent_type": fallback_agent.get("agent_type", "unknown"),
                            "content": content,
                            "stance": stance,
                            "score": 0,
                            "likes": 0,
                            "unlikes": 0,
                            "author_id": fallback_id,
                            "belief": beliefs[fallback_id],
                            "author_influence": self._get_authority_score(fallback_agent),
                        }
                    )
                    self.forum_posts[fallback_id] = post_id

        self.step_posts[step] = posts
        return posts

    def _should_publish(self, agent, current_belief):
        if not agent.get("is_active", True) and random.random() > 0.15:
            return False
        initial_belief = float(agent.get("initial_belief", 0.0))
        activity_level = float(agent.get("activity_level", 0.5))
        expression_threshold = float(agent.get("expression_threshold", 0.2))
        belief_shift = abs(current_belief - initial_belief)
        return belief_shift >= expression_threshold or random.random() < activity_level

    def _compose_post_content(self, agent):
        reaction = agent.get("reaction") or {}
        understanding = reaction.get("understanding", "").strip()
        recommendation = reaction.get("recommendation", "").strip()
        concerns = reaction.get("concerns_analysis", "").strip()

        segments = [segment for segment in [understanding, recommendation, concerns] if segment]
        content = "；".join(segments[:2]).strip()
        return content[:200]

    def _get_exposure_pool(self, agent_id, step_posts, top_posts):
        neighbor_quota = min(3, self.EXPOSURE_LIMIT)
        authority_quota = 1 if self.EXPOSURE_LIMIT > 1 else 0
        hot_quota = 1 if self.EXPOSURE_LIMIT > 2 else 0

        neighbor_ids = set(self.graph.get(agent_id, {}).keys())
        neighbor_posts = [
            post
            for post in step_posts
            if post["author_id"] in neighbor_ids and post["author_id"] != agent_id
        ]
        authority_posts = sorted(
            [post for post in step_posts if post["author_id"] != agent_id],
            key=lambda post: post.get("author_influence", 0.5),
            reverse=True,
        )
        hot_posts = [
            self._decorate_hot_post(post)
            for post in top_posts
            if post.get("agent_id") != agent_id
        ]

        selected = []
        selected_ids = set()
        self._select_posts(selected, selected_ids, neighbor_posts, neighbor_quota)
        self._select_posts(selected, selected_ids, authority_posts, authority_quota)
        self._select_posts(selected, selected_ids, hot_posts, hot_quota)

        if len(selected) < self.EXPOSURE_LIMIT:
            fallback_pool = authority_posts + neighbor_posts + hot_posts
            self._select_posts(
                selected, selected_ids, fallback_pool, self.EXPOSURE_LIMIT - len(selected)
            )
        return selected[: self.EXPOSURE_LIMIT]

    def _select_posts(self, selected, selected_ids, pool, quota):
        for post in pool:
            if len(selected) >= self.EXPOSURE_LIMIT or quota <= 0:
                break
            if post["id"] in selected_ids:
                continue
            selected.append(post)
            selected_ids.add(post["id"])
            quota -= 1

    def _decorate_hot_post(self, post):
        decorated = dict(post)
        author_id = post.get("agent_id")
        decorated["author_id"] = author_id
        if author_id in self.agent_by_id:
            author_agent = self.agent_by_id[author_id]
            decorated["belief"] = float(author_agent.get("belief", 0.0))
            decorated["author_influence"] = self._get_authority_score(author_agent)
        else:
            decorated["belief"] = self.STANCE_TO_BELIEF.get(post.get("stance"), 0.0)
            decorated["author_influence"] = 0.5
        return decorated

    def _calculate_forum_influence(self, agent_id, agent, exposure_pool, beliefs):
        if not exposure_pool:
            return beliefs[agent_id], {}, 0, {
                "exposed_post_ids": [],
                "adopted_post_ids": [],
            }

        current_belief = beliefs[agent_id]
        susceptibility = float(agent.get("susceptibility", 0.5))
        repost_tendency = float(agent.get("repost_tendency", 0.2))

        interaction_signals = {}
        weighted_sum = 0.0
        total_weight = 0.0
        adopted_count = 0
        adopted_post_ids = []
        exposed_post_ids = [post["id"] for post in exposure_pool]

        for post in exposure_pool:
            author_id = post.get("author_id")
            if author_id is None or author_id == agent_id:
                continue

            post_belief = float(
                post.get("belief", self.STANCE_TO_BELIEF.get(post.get("stance"), 0.0))
            )
            alignment = max(0.0, 1.0 - abs(current_belief - post_belief) / 2.0)
            author_influence = float(post.get("author_influence", 0.5))

            if alignment >= 0.72:
                like_probability = min(
                    0.95, 0.35 + 0.40 * alignment + 0.25 * susceptibility
                )
                if random.random() < like_probability:
                    self.forum_service.react_to_post(agent_id, post["id"], "like")
                    amplification = 1.0
                    if alignment >= 0.88 and random.random() < repost_tendency:
                        amplification += 0.15

                    adoption_weight = alignment * author_influence * susceptibility
                    adoption_weight *= amplification
                    weighted_sum += post_belief * adoption_weight
                    total_weight += adoption_weight
                    interaction_signals[author_id] = (
                        interaction_signals.get(author_id, 0.0) + amplification
                    )
                    adopted_count += 1
                    adopted_post_ids.append(post["id"])
                    continue

            if alignment <= 0.25:
                dislike_probability = min(
                    0.85, 0.20 + 0.45 * (1 - alignment) * (0.5 + susceptibility / 2)
                )
                if random.random() < dislike_probability:
                    self.forum_service.react_to_post(agent_id, post["id"], "unlike")
                    interaction_signals[author_id] = (
                        interaction_signals.get(author_id, 0.0) - 1.0
                    )

        if total_weight <= 0:
            return current_belief, interaction_signals, adopted_count, {
                "exposed_post_ids": exposed_post_ids,
                "adopted_post_ids": adopted_post_ids,
            }
        return weighted_sum / total_weight, interaction_signals, adopted_count, {
            "exposed_post_ids": exposed_post_ids,
            "adopted_post_ids": adopted_post_ids,
        }

    def _calculate_social_influence(self, agent_id, beliefs):
        neighbors = self.graph.get(agent_id, {})
        if not neighbors:
            return beliefs[agent_id]

        weighted_sum = 0.0
        total_weight = 0.0
        for neighbor_id, base_weight in neighbors.items():
            difference = abs(beliefs[agent_id] - beliefs[neighbor_id])
            confidence_multiplier = (
                1.0
                if difference <= self.CONFIDENCE_THRESHOLD
                else self.CONFIDENCE_DECAY
            )
            effective_weight = base_weight * confidence_multiplier
            weighted_sum += beliefs[neighbor_id] * effective_weight
            total_weight += effective_weight

        if total_weight <= 0:
            return beliefs[agent_id]
        return weighted_sum / total_weight

    def _calculate_forum_weight(self, agent):
        activity_level = float(agent.get("activity_level", 0.5))
        profile = agent.get("profile", {}) or {}
        network_influence = float(profile.get("network_influence", 0.5))
        forum_weight = 0.1 + 0.25 * activity_level + 0.15 * network_influence
        return self._clamp(forum_weight, 0.15, 0.45)

    def _update_edge_weights(self, interaction_signals, beliefs):
        top_k = min(self.DEFAULT_TOP_K, len(self.agents) - 1) if len(self.agents) > 1 else 0
        updated_graph = {}

        for source_agent in self.agents:
            source_id = source_agent["id"]
            outgoing = dict(self.graph.get(source_id, {}))
            for target_id, signal in interaction_signals.get(source_id, {}).items():
                if target_id == source_id:
                    continue
                outgoing.setdefault(target_id, self.MIN_EDGE_WEIGHT)
                if signal > 0:
                    outgoing[target_id] += self.REINFORCE_DELTA * signal
                elif signal < 0:
                    outgoing[target_id] += self.WEAKEN_DELTA * signal

            for target_id in list(outgoing.keys()):
                divergence = abs(beliefs[source_id] - beliefs[target_id])
                if divergence > self.CONFIDENCE_THRESHOLD * 1.4:
                    outgoing[target_id] = max(outgoing[target_id] - 0.02, 0.0)

            filtered = {
                id: weight
                for id, weight in outgoing.items()
                if id != source_id and weight >= self.MIN_EDGE_WEIGHT
            }

            if not filtered and len(self.agents) > 1:
                candidates = []
                for target_agent in self.agents:
                    if target_agent["id"] == source_id:
                        continue
                    base_weight = self._compute_edge_weight(
                        source_agent, target_agent
                    )
                    candidates.append((target_agent["id"], base_weight))
                candidates.sort(key=lambda item: item[1], reverse=True)
                if candidates:
                    id, weight = candidates[0]
                    filtered[id] = max(weight, self.MIN_EDGE_WEIGHT)

            top_edges = sorted(
                filtered.items(), key=lambda item: item[1], reverse=True
            )[:top_k]
            total_weight = sum(weight for _, weight in top_edges)
            if total_weight <= 0:
                updated_graph[source_id] = {}
                continue

            updated_graph[source_id] = {
                id: weight / total_weight for id, weight in top_edges
            }

        self.graph = updated_graph

    def _get_social_neighbor_trace(self, agent_id):
        neighbors = sorted(
            self.graph.get(agent_id, {}).items(), key=lambda item: item[1], reverse=True
        )[:3]
        trace = []
        for neighbor_id, weight in neighbors:
            neighbor_agent = self.agent_by_id.get(neighbor_id, {})
            trace.append(
                {
                    "agent_id": neighbor_id,
                    "role": neighbor_agent.get("role", "unknown"),
                    "weight": round(weight, 4),
                }
            )
        return trace

    def _augment_forum_stats(self, stats):
        stats = stats or {}
        stats["behavior_metrics"] = {
            "active_posters_count": self.latest_step_metrics.get(
                "active_posters_count", 0
            ),
            "forum_exposed_count": self.latest_step_metrics.get(
                "forum_exposed_count", 0
            ),
        }
        return stats

    def _get_network_data(
        self,
        beliefs,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        statistics: Optional[Dict[str, Any]] = None,
    ):
        nodes = []
        influence_scores = self._calculate_influence_scores()
        positions = self._build_layout_positions(influence_scores)
        for agent in self.agents:
            agent_id = agent["id"]
            initial_belief = float(agent.get("initial_belief", beliefs[agent_id]))
            current_belief = round(beliefs[agent_id], 3)
            profile = agent.get("profile", {})
            display_label = self._build_agent_label(agent)
            position = positions.get(agent_id)
            if position is None:
                position = self._calculate_position(len(nodes), len(self.agents))
            nodes.append(
                {
                    "id": agent_id,
                    "role": agent["role"],
                    "role_name": agent.get("role_name", agent["role"]),
                    "agent_type": agent.get("agent_type", "unknown"),
                    "label": display_label,
                    "color": agent["color"],
                    "belief": current_belief,
                    "initial_belief": round(initial_belief, 3),
                    "belief_shift": round(current_belief - initial_belief, 3),
                    "position": position,
                    "influence_score": round(influence_scores.get(agent_id, 0.0), 3),
                    "activity_level": round(float(agent.get("activity_level", 0.5)), 3),
                    "stubbornness": round(float(agent.get("stubbornness", 0.5)), 3),
                    "reputation_score": round(
                        self._safe_float(profile.get("reputation_score"), 0.5), 3
                    ),
                    "network_influence": round(
                        self._safe_float(profile.get("network_influence"), 0.5), 3
                    ),
                }
            )

        edges = []
        for source_id, neighbors in self.graph.items():
            for target_id, weight in sorted(
                neighbors.items(), key=lambda item: item[1], reverse=True
            ):
                source_agent = self.agent_by_id.get(source_id, {})
                target_agent = self.agent_by_id.get(target_id, {})
                edges.append(
                    {
                        "id": f"{source_id}->{target_id}",
                        "source": source_id,
                        "target": target_id,
                        "weight": round(weight, 3),
                        "cross_role": source_agent.get("role", "")
                        != target_agent.get("role", ""),
                    }
                )

        role_counts = defaultdict(int)
        for agent in self.agents:
            role_counts[agent["role"]] += 1

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "step": step if step is not None else 0,
                "total_steps": total_steps if total_steps is not None else 1,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "role_counts": dict(role_counts),
                "statistics": statistics or {},
            },
        }

    def _calculate_influence_scores(self):
        incoming = defaultdict(float)
        for _, neighbors in self.graph.items():
            for target_id, weight in neighbors.items():
                incoming[target_id] += weight

        scores = {}
        for agent in self.agents:
            agent_id = agent["id"]
            scores[agent_id] = self._clamp(
                incoming.get(agent_id, 0.0) * 0.6 + self._get_authority_score(agent) * 0.4,
                0.0,
                1.0,
            )
        return scores

    def _calculate_position(self, index, total):
        angle = (2 * math.pi * index / max(total, 1)) - (math.pi / 2)
        radius = 0.34

        return {
            "x": round(0.5 + radius * math.cos(angle), 4),
            "y": round(0.5 + radius * math.sin(angle), 4),
        }

    def _build_layout_positions(self, influence_scores):
        grouped_ids = defaultdict(list)
        for agent in self.agents:
            grouped_ids[agent["role"]].append(agent["id"])

        ordered_roles = [
            role for role in self.ROLE_LAYOUT_ORDER if grouped_ids.get(role)
        ]
        ordered_roles.extend(
            role for role in grouped_ids.keys() if role not in ordered_roles
        )

        positions = {}
        total_roles = max(len(ordered_roles), 1)
        for role_order, role in enumerate(ordered_roles):
            anchor = self.ROLE_CLUSTER_POSITIONS.get(
                role, self._calculate_position(role_order, total_roles)
            )
            ranked_ids = sorted(
                grouped_ids[role],
                key=lambda agent_id: (
                    -influence_scores.get(agent_id, 0.0),
                    agent_id
                ),
            )
            for member_order, agent_id in enumerate(ranked_ids):
                positions[agent_id] = self._calculate_cluster_position(
                    anchor,
                    member_order,
                    influence_scores.get(agent_id, 0.0),
                    role_order,
                )

        return positions

    def _calculate_cluster_position(
        self,
        anchor: Dict[str, float],
        member_order: int,
        influence_score: float,
        role_order: int,
    ) -> Dict[str, float]:
        if member_order == 0:
            return {
                "x": round(self._clamp(anchor["x"], 0.08, 0.92), 4),
                "y": round(self._clamp(anchor["y"], 0.10, 0.92), 4),
            }

        layer = (member_order - 1) // 6
        slot = (member_order - 1) % 6
        base_radius = 0.072 + layer * 0.038
        radius = base_radius * (1.15 - 0.35 * influence_score)
        angle = (2 * math.pi * slot / 6) + role_order * 0.35

        return {
            "x": round(
                self._clamp(anchor["x"] + radius * math.cos(angle), 0.08, 0.92), 4
            ),
            "y": round(
                self._clamp(anchor["y"] + radius * math.sin(angle), 0.10, 0.92), 4
            ),
        }

    def _build_agent_label(self, agent: Dict[str, Any]) -> str:
        role_name = str(agent.get("role_name") or agent.get("role") or "角色")
        match = re.search(r"(\d+)$", str(agent.get("id", "")))
        if not match:
            return role_name
        return f"{role_name}{int(match.group(1))}"

    def _calculate_statistics(self, beliefs):
        belief_values = list(beliefs.values())
        if not belief_values:
            return {}

        count = len(belief_values)
        mean = sum(belief_values) / count
        variance = sum((belief - mean) ** 2 for belief in belief_values) / count
        polarization = math.sqrt(variance)

        conservative = sum(1 for belief in belief_values if belief < -0.2)
        neutral = sum(1 for belief in belief_values if -0.2 <= belief <= 0.2)
        aggressive = sum(1 for belief in belief_values if belief > 0.2)

        edge_count = sum(len(neighbors) for neighbors in self.graph.values())
        avg_out_degree = edge_count / count if count else 0.0
        incoming = defaultdict(int)
        cross_role_weight = 0.0
        total_weight = 0.0
        for source_id, neighbors in self.graph.items():
            for target_id, weight in neighbors.items():
                incoming[target_id] += 1
                total_weight += weight
                source_agent = self.agent_by_id.get(source_id, {})
                target_agent = self.agent_by_id.get(target_id, {})
                if source_agent.get("role", "") != target_agent.get("role", ""):
                    cross_role_weight += weight

        avg_in_degree = sum(incoming.values()) / count if count else 0.0
        cross_role_ratio = cross_role_weight / total_weight if total_weight else 0.0

        belief_shifts = [
            abs(float(agent.get("initial_belief", 0.0)) - beliefs[agent["id"]])
            for agent in self.agents
        ]

        return {
            "mean_belief": round(mean, 3),
            "variance": round(variance, 3),
            "polarization_index": round(polarization, 3),
            "conservative_ratio": round(conservative / count, 3),
            "neutral_ratio": round(neutral / count, 3),
            "aggressive_ratio": round(aggressive / count, 3),
            "conservative_count": conservative,
            "neutral_count": neutral,
            "aggressive_count": aggressive,
            "avg_out_degree": round(avg_out_degree, 3),
            "avg_in_degree": round(avg_in_degree, 3),
            "active_posters_count": self.latest_step_metrics.get(
                "active_posters_count", 0
            ),
            "forum_exposed_count": self.latest_step_metrics.get(
                "forum_exposed_count", 0
            ),
            "cross_role_influence_ratio": round(cross_role_ratio, 3),
            "belief_shift_mean": round(sum(belief_shifts) / count, 3),
            "belief_shift_max": round(max(belief_shifts), 3),
        }

    def _calculate_polarization(self, beliefs):
        stats = self._calculate_statistics(beliefs)
        return stats.get("polarization_index", 0.0)

    def get_network_structure(self):
        node_count = len(self.agents)
        edge_count = sum(len(neighbors) for neighbors in self.graph.values())

        if node_count == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "average_degree": 0.0,
                "average_out_degree": 0.0,
                "average_in_degree": 0.0,
                "average_clustering": 0.0,
            }

        incoming = defaultdict(int)
        undirected_adj = defaultdict(set)
        for source_id, neighbors in self.graph.items():
            for target_id in neighbors:
                incoming[target_id] += 1
                undirected_adj[source_id].add(target_id)
                undirected_adj[target_id].add(source_id)

        avg_out_degree = edge_count / node_count
        avg_in_degree = sum(incoming.values()) / node_count

        clustering_coefficients = []
        for agent_id in self.agent_by_id.keys():
            neighbors = list(undirected_adj[agent_id])
            if len(neighbors) < 2:
                continue
            possible_links = len(neighbors) * (len(neighbors) - 1) / 2
            actual_links = 0
            for i, neighbor_a in enumerate(neighbors):
                for neighbor_b in neighbors[i + 1 :]:
                    if neighbor_b in undirected_adj[neighbor_a]:
                        actual_links += 1
            clustering_coefficients.append(actual_links / possible_links)

        avg_clustering = (
            sum(clustering_coefficients) / len(clustering_coefficients)
            if clustering_coefficients
            else 0.0
        )

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "average_degree": round((avg_out_degree + avg_in_degree) / 2, 3),
            "average_out_degree": round(avg_out_degree, 3),
            "average_in_degree": round(avg_in_degree, 3),
            "average_clustering": round(avg_clustering, 3),
        }

    def _belief_to_stance(self, belief):
        """统一阈值：与 ReactionService.decision_stance 计算保持一致（±0.15）"""
        if belief < -0.15:
            return "保守"
        if belief > 0.15:
            return "激进"
        return "中立"

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value, minimum, maximum):
        return max(minimum, min(maximum, value))
