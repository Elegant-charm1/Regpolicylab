from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import PolicyAgent, ReactionResult
from util.PolicyForumDB import (
    create_post,
    get_all_posts,
    get_forum_statistics,
    get_top_posts,
    init_policy_forum,
    react_to_post,
)


class ForumService:
    def __init__(self, session_id: str, db_path: Optional[str] = None) -> None:
        self.session_id = session_id
        self.db_path = db_path or self.build_db_path(session_id)

    @staticmethod
    def build_db_path(session_id: str) -> str:
        safe_session_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        project_root = Path(__file__).resolve().parent.parent
        return str(project_root / "data" / f"policy_forum_{safe_session_id}.db")

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
            agent_id,
            agent_role,
            agent_type,
            content,
            stance,
            step,
            self.db_path,
        )

    def react_to_post(self, agent_id: str, post_id: int, reaction_type: str) -> bool:
        return react_to_post(agent_id, post_id, reaction_type, self.db_path)

    def get_top_posts(self, step: int, limit: int = 10) -> List[Dict[str, Any]]:
        return get_top_posts(step, limit, self.db_path)

    def get_all_posts(self) -> List[Dict[str, Any]]:
        return get_all_posts(self.db_path)

    def get_forum_statistics(self, step: int) -> Dict[str, Any]:
        return get_forum_statistics(step, self.db_path)


class ReactionPromptBuilder:
    ROLE_REALISM = {
        "accountant": {
            "accountability": "会计处理是否有准则依据、口径是否一致、披露是否经得起审计和监管复核",
            "primary_goal": "在可执行的前提下保证会计处理合规、稳妥、可落地",
            "red_lines": "没有准则依据就贸然确认、披露口径前后不一致、给后续审计留明显漏洞",
            "tone": "专业、克制、证据导向",
        },
        "auditor": {
            "accountability": "审计证据是否充分、风险识别是否到位、结论是否站得住脚",
            "primary_goal": "降低审计失败和错报漏报风险，优先守住证据与程序底线",
            "red_lines": "证据不足却接受处理、对高风险事项轻描淡写、无法向项目质量复核交代",
            "tone": "怀疑、谨慎、程序导向",
        },
        "manager": {
            "accountability": "公司经营稳定性、资本市场预期、声誉与执行成本",
            "primary_goal": "在合规前提下平衡业绩、融资、声誉和资源投入",
            "red_lines": "引发重大合规事件、财务指标剧烈波动失控、对外沟通失真",
            "tone": "平衡、多方权衡、结果导向",
        },
        "regulator": {
            "accountability": "政策是否被正确理解、市场秩序是否稳定、执行口径是否一致",
            "primary_goal": "推动政策有效落地，同时控制误解、套利和执行偏差",
            "red_lines": "政策口径模糊导致市场误判、执行选择性失衡、监管公信力受损",
            "tone": "权威、克制、强调执行效果",
        },
        "investor": {
            "accountability": "仓位决策、回撤控制、信息解读是否影响收益与风险",
            "primary_goal": "判断政策对估值、预期和交易时机的真实影响",
            "red_lines": "依据不充分就重仓行动、忽视流动性和波动风险、误判政策方向",
            "tone": "直接、收益风险并重",
        },
    }

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _belief_label(self, value: float) -> str:
        if value <= -0.35:
            return "明显谨慎"
        if value < -0.10:
            return "偏谨慎"
        if value >= 0.35:
            return "明显积极"
        if value > 0.10:
            return "偏积极"
        return "观望中性"

    def _relevance_label(self, relevance_type: str, value: float) -> str:
        if relevance_type == "direct" or value >= 0.75:
            return "直接相关"
        if relevance_type == "adjacent" or value >= 0.45:
            return "间接相关"
        return "外围关注"

    def _activation_label(self, activation_tier: str) -> str:
        return {
            "lead": "主动发声者",
            "react": "跟进反应者",
            "observe": "低活跃观察者",
        }.get(activation_tier, "跟进反应者")

    def _build_realism_brief(self, agent: PolicyAgent) -> str:
        profile = agent.profile or {}
        org_context = agent.org_context or {}
        scenario_context = agent.scenario_context or {}
        characteristics = agent.characteristics or {}
        realism = self.ROLE_REALISM.get(agent.role, self.ROLE_REALISM["accountant"])

        name = profile.get("name") or agent.id
        organization = org_context.get("organization_type") or profile.get(
            "organization_type"
        ) or "所在机构"
        specialization = profile.get("specialization") or "本职领域"
        industry_focus = profile.get("industry_focus") or "相关行业"
        risk_attitude = profile.get("risk_attitude") or "中性"
        decision_style = profile.get("decision_style") or "常规判断"
        experience_years = profile.get("experience_years") or "未知"
        policy_sensitivity = self._safe_float(
            characteristics.get("policy_sensitivity"), 0.6
        )
        conservative_bias = self._safe_float(
            characteristics.get("conservative_bias"), 0.5
        )
        expert_trust = self._safe_float(characteristics.get("expert_trust"), 0.5)
        policy_relevance = self._safe_float(agent.policy_relevance, 0.5)
        prior_belief = self._safe_float(agent.prior_belief, 0.0)
        enforcement_pressure = self._safe_float(agent.enforcement_pressure, 0.5)
        public_attention = self._safe_float(
            scenario_context.get("public_attention"), 0.5
        )
        disclosure_pressure = self._safe_float(
            scenario_context.get("disclosure_pressure"), 0.5
        )
        policy_domain = scenario_context.get("policy_domain") or agent.affected_domain
        relevance_label = self._relevance_label(agent.relevance_type, policy_relevance)
        prior_label = self._belief_label(prior_belief)
        activation_label = self._activation_label(agent.activation_tier)
        influence_tier = agent.influence_tier or "medium"
        voice_style = agent.voice_style or realism["tone"]
        organization_size = org_context.get("organization_size") or "medium"
        decision_authority = org_context.get("decision_authority") or "medium"
        accountability_pressure = org_context.get("accountability_pressure") or "medium"

        direct_drivers = "、".join(agent.concerns[:4]) if agent.concerns else "合规与执行"

        return f"""
## 你的身份锚点
- 你不是泛化专家，而是现实中的具体从业者：{name}
- 所在机构：{organization}
- 主要职责：{specialization}
- 关注行业/场景：{industry_focus}
- 工作经验：{experience_years}年
- 风险态度：{risk_attitude}
- 决策风格：{decision_style}

## 你的真实约束
- 你要为以下结果负责：{realism['accountability']}
- 你当前最优先保护的是：{realism['primary_goal']}
- 你最不能接受的是：{realism['red_lines']}
- 你的组织规模：{organization_size}
- 你的决策权限：{decision_authority}
- 你的问责压力：{accountability_pressure}
- 你天然会重点盯住：{direct_drivers}
- 你对政策变化的敏感度：{policy_sensitivity:.2f}
- 你的保守倾向：{conservative_bias:.2f}
- 你对专业判断可信度的要求：{expert_trust:.2f}

## 这轮政策场景
- 政策领域：{policy_domain}
- 你与该政策的关系：{relevance_label}（相关度 {policy_relevance:.2f}）
- 你当前的先验立场：{prior_label}（prior_belief={prior_belief:.2f}）
- 你在本轮讨论中的位置：{activation_label}
- 你的话语权档位：{influence_tier}
- 执行压力：{enforcement_pressure:.2f}
- 公众关注度：{public_attention:.2f}
- 披露压力：{disclosure_pressure:.2f}

## 表达要求
- 请用 {voice_style} 的方式回答
- 如果政策与自身职责只有间接关系，要明确说清“间接影响”，不要夸大
- 不要写成泛泛的专家评论，要写成一个真的会被追责、会被问责、要承担后果的人
""".strip()

    def build(self, agent: PolicyAgent, policy_content: str) -> str:
        concerns = ", ".join(agent.concerns)
        realism_brief = self._build_realism_brief(agent)
        return f"""
# 任务说明
请以{agent.role_name}的身份，分析以下政策/新闻公告对你的影响，并给出专业判断。

---

## 政策/新闻内容
{policy_content}

---

## 你的角色设定
{agent.system_prompt or ('你是一位专业的' + agent.role_name)}

---

{realism_brief}

---

## 思考步骤（Chain of Thought）
请按照以下步骤逐步分析，这将帮助你形成更全面、专业的判断：

**步骤0：相关性判断**
- 先判断该政策和你的真实职责是直接相关、间接相关，还是仅需关注
- 如果只是间接相关，不要假装自己是政策核心执行者
- 你的初始立场不是空白的，请基于给定的 prior_belief 和组织约束进入讨论

**步骤1：政策理解**
- 识别政策的核心内容和关键变化点
- 分析政策涉及的具体会计处理或监管要求

**步骤2：影响评估**
- 从你的角色角度评估政策对你的主要影响
- 识别影响程度（高/中/低）和影响范围
- 明确这会影响你的哪个具体工作结果、考核目标或问责风险

**步骤3：风险分析**
- 分析你最关注的风险点（{concerns}）
- 评估潜在的不确定性和应对难度
- 说明你最不能接受的后果是什么，以及为什么

**步骤4：决策形成**
- 综合以上分析，确定你的决策倾向
- 明确具体的行动建议或应对方案
- 建议必须符合你的职位权限、组织约束和专业身份，不能说空话

---

## 输出要求
请以严格的JSON格式输出你的分析结果（不要添加任何额外说明文字）：

```json
{{
    "understanding": "你对政策核心内容的理解（200字以内，详细阐述关键要点，包括政策意图、核心变化、适用范围）",
    "decision_stance": "你的决策倾向（必须是以下之一：保守/中立/激进）",
    "impact_level": "影响程度评估（必须是以下之一：高/中/低）",
    "concerns_analysis": "你最关注的方面分析（详细说明你最担心的风险点及其原因）",
    "recommendation": "你的具体建议或行动方案（给出可执行的具体建议，而非笼统描述）",
    "decision_reason": "做出此决策的核心理由（结合你的角色特征和风险偏好进行说明）",
    "belief_score": 你的信念值（数值，范围-1.0到1.0，负数表示保守倾向，正数表示激进倾向，0表示中立）
}}
```

**重要提示：**
- belief_score 必须是纯数值，不要加引号
- 所有字符串字段必须用引号包裹
- 确保 JSON 格式完全正确，可直接解析
- recommendation 和 concerns_analysis 要具体，避免空洞表述
""".strip()


class ReactionParser:
    REQUIRED_STRING_FIELDS = [
        "understanding",
        "decision_stance",
        "impact_level",
        "concerns_analysis",
        "recommendation",
        "decision_reason",
    ]

    def parse(self, content: str, fallback: ReactionResult) -> ReactionResult:
        reaction_data = self._extract_json_dict(content)
        normalized = fallback.to_dict()
        normalized.update(reaction_data)

        for field in self.REQUIRED_STRING_FIELDS:
            value = normalized.get(field, getattr(fallback, field))
            if isinstance(value, dict):
                value = value.get("content") or value.get("summary") or str(value)
            normalized[field] = str(value)

        try:
            belief_score = float(normalized.get("belief_score", fallback.belief_score))
        except (TypeError, ValueError):
            belief_score = fallback.belief_score
        normalized["belief_score"] = max(-1.0, min(1.0, belief_score))

        return ReactionResult.from_dict(normalized)

    def _extract_json_dict(self, content: str) -> Dict[str, Any]:
        json_candidates = []

        fenced_json = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if fenced_json:
            json_candidates.append(fenced_json.group(1))

        fenced = re.search(r"```\s*([\s\S]*?)\s*```", content)
        if fenced:
            json_candidates.append(fenced.group(1))

        braced = re.search(r"\{[\s\S]*\}", content)
        if braced:
            json_candidates.append(braced.group(0))

        json_candidates.append(content)

        for candidate in json_candidates:
            cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", candidate).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}


class ReactionService:
    def __init__(
        self,
        prompt_builder: Optional[ReactionPromptBuilder] = None,
        parser: Optional[ReactionParser] = None,
    ) -> None:
        self.prompt_builder = prompt_builder or ReactionPromptBuilder()
        self.parser = parser or ReactionParser()

    def generate_reactions(
        self, agents: List[PolicyAgent], policy_content: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        reactions = []
        for agent in agents:
            reaction = self.generate_single_reaction(agent, policy_content)
            agent.apply_reaction(reaction)
            reactions.append(self.serialize_agent_profile(agent))

        return reactions, self.calculate_statistics(reactions)

    def generate_single_reaction(
        self, agent: PolicyAgent, policy_content: str
    ) -> ReactionResult:
        fallback = self._get_default_reaction(agent)
        llm_agent = agent.llm_agent
        if llm_agent is None:
            return fallback

        prompt = self.prompt_builder.build(agent, policy_content)
        try:
            response = llm_agent.get_response(
                user_input=prompt,
                temperature=0.7,
                max_tokens=1024,
            )
        except Exception as exc:
            return self._get_default_reaction(agent, str(exc))

        if "error" in response:
            return self._get_default_reaction(agent, str(response["error"]))

        content = response.get("response", "")
        return self.parser.parse(content, fallback)

    def serialize_agent_profile(self, agent: PolicyAgent) -> Dict[str, Any]:
        return agent.to_public_dict()

    def calculate_statistics(self, reactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not reactions:
            return {}

        stance_counts = {"保守": 0, "中立": 0, "激进": 0}
        role_stats: Dict[str, Dict[str, float]] = {}

        for reaction in reactions:
            stance = reaction.get("reaction", {}).get("decision_stance", "中立")
            stance_counts[stance] = stance_counts.get(stance, 0) + 1

            role = reaction["role"]
            role_stats.setdefault(role, {"count": 0, "avg_belief": 0.0})
            role_stats[role]["count"] += 1
            role_stats[role]["avg_belief"] += float(reaction.get("belief", 0.0))

        for role, stats in role_stats.items():
            if stats["count"]:
                stats["avg_belief"] /= stats["count"]

        total_belief = sum(float(item.get("belief", 0.0)) for item in reactions)
        avg_belief = total_belief / len(reactions) if reactions else 0.0

        return {
            "stance_distribution": stance_counts,
            "role_statistics": role_stats,
            "average_belief": avg_belief,
            "total_reactions": len(reactions),
        }

    def _get_default_reaction(
        self, agent: PolicyAgent, error_msg: str = ""
    ) -> ReactionResult:
        if error_msg:
            print(f"[调试] {agent.id} 使用默认反应: {error_msg}")
        return self._build_realistic_default_reaction(agent)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _build_realistic_default_reaction(self, agent: PolicyAgent) -> ReactionResult:
        profile = agent.profile or {}
        org_context = agent.org_context or {}
        scenario_context = agent.scenario_context or {}
        characteristics = agent.characteristics or {}
        organization = org_context.get("organization_type") or profile.get(
            "organization_type"
        ) or "所在机构"
        specialization = profile.get("specialization") or agent.role_name
        decision_style = profile.get("decision_style") or "常规判断"
        risk_attitude = profile.get("risk_attitude") or "中性"
        policy_domain = scenario_context.get("policy_domain") or agent.affected_domain
        relevance_type = agent.relevance_type or "ambient"
        policy_relevance = self._safe_float(agent.policy_relevance, 0.5)
        prior_belief = self._safe_float(agent.prior_belief, 0.0)
        public_attention = self._safe_float(
            scenario_context.get("public_attention"), 0.5
        )
        decision_authority = org_context.get("decision_authority") or "medium"
        accountability_pressure = (
            org_context.get("accountability_pressure") or "medium"
        )
        key_concerns = (
            "、".join(agent.concerns[:3]) if agent.concerns else "合规、执行、影响评估"
        )

        activation_adjustment = {
            "lead": 0.08,
            "react": 0.00,
            "observe": -0.05,
        }.get(agent.activation_tier, 0.0)
        belief_score = self._clamp(
            prior_belief + activation_adjustment + (policy_relevance - 0.5) * 0.12,
            -1.0,
            1.0,
        )

        if belief_score < -0.15:
            decision_stance = "保守"
        elif belief_score > 0.15:
            decision_stance = "激进"
        else:
            decision_stance = "中立"

        if policy_relevance >= 0.72:
            impact_level = "高"
        elif policy_relevance >= 0.45:
            impact_level = "中"
        else:
            impact_level = "低"

        relation_text = {
            "direct": "直接承担执行或应对责任",
            "adjacent": "会受到明显的连带影响",
            "ambient": "更多需要持续关注边际变化",
        }.get(relevance_type, "会受到一定影响")

        role_understanding = {
            "accountant": f"作为{organization}中负责{specialization}的会计人员，我会把该政策理解为{policy_domain}领域的一次口径变化，它对我属于{relation_text}。",
            "auditor": f"作为{organization}中负责{specialization}的审计人员，我会优先把该政策理解为{policy_domain}领域的风险重分配，它对我属于{relation_text}。",
            "manager": f"作为{organization}中的管理者，我会把该政策理解为会影响经营结果、市场沟通和资源投入的一次{policy_domain}约束，它对我属于{relation_text}。",
            "regulator": f"作为{organization}中负责{specialization}的监管人员，我会将该政策理解为需要统一执行口径并稳定预期的{policy_domain}动作，它对我属于{relation_text}。",
            "investor": f"作为关注{specialization}的市场参与者，我会把该政策理解为可能改变信息质量、估值预期和交易节奏的{policy_domain}变量，它对我属于{relation_text}。",
        }

        role_recommendation = {
            "accountant": "先做口径梳理、差异比对和披露清单校验，再决定是否调整会计处理。",
            "auditor": "优先补强证据链、重新评估高风险科目，并提高对管理层解释的一致性核查强度。",
            "manager": "先评估对利润、披露节奏和外部沟通的影响，再决定推进速度和对外表述口径。",
            "regulator": "应尽快明确执行口径、发布解读重点，并把高风险误用场景纳入重点观察。",
            "investor": "先观察市场如何重估相关公司信息质量，再决定是否调整仓位和行业暴露。",
        }
        if agent.activation_tier == "observe":
            role_recommendation[agent.role] = "先持续跟踪政策口径和市场反馈，暂不做超出职责边界的激进行动。"
        elif agent.activation_tier == "lead":
            role_recommendation[agent.role] = {
                "accountant": "尽快牵头梳理内部执行口径和披露清单，形成可落地的处理方案。",
                "auditor": "尽快明确重点风险和补充程序要求，主动推动高风险事项复核。",
                "manager": "尽快组织财务、法务和投资者沟通团队形成统一对外口径。",
                "regulator": "尽快给出执行口径和边界说明，压缩市场误读空间。",
                "investor": "优先围绕高相关资产做情景推演，准备仓位和交易节奏调整方案。",
            }.get(agent.role, role_recommendation[agent.role])

        role_reason = {
            "accountant": "我的首要约束是处理口径是否站得住、后续审计是否能通过，而不是追求表面上的灵活性。",
            "auditor": "我的判断必须能够经得起项目复核和责任追溯，因此会天然偏向证据充分和程序稳妥。",
            "manager": "我需要同时对经营结果、市场预期和执行成本负责，所以不会只从单一合规视角看问题。",
            "regulator": "我必须优先考虑政策执行的一致性、市场秩序和监管公信力，因此更关注落地效果。",
            "investor": "我最终要对收益风险比负责，因此更在意政策是否会改变预期差和估值锚。",
        }

        concerns_analysis = (
            f"我会优先关注{key_concerns}。"
            f"在{organization}这一现实约束下，{decision_style}意味着我不能忽视{risk_attitude}所对应的问责风险。"
            f" 当前我的相关度为{policy_relevance:.2f}，决策权限为{decision_authority}，问责压力为{accountability_pressure}。"
        )

        recommendation = role_recommendation.get(agent.role, role_recommendation["accountant"])
        decision_reason = (
            f"{role_reason.get(agent.role, role_reason['accountant'])}"
            f" 当前我的先验立场为{prior_belief:.2f}，且本轮公众关注度为{public_attention:.2f}，因此会采取与职责边界一致的判断。"
        )

        return ReactionResult(
            understanding=role_understanding.get(
                agent.role, role_understanding["accountant"]
            ),
            decision_stance=decision_stance,
            impact_level=impact_level,
            concerns_analysis=concerns_analysis,
            recommendation=recommendation,
            decision_reason=decision_reason,
            belief_score=round(belief_score, 3),
        )


class PropagationService:
    def __init__(self, forum_service: ForumService) -> None:
        self.forum_service = forum_service

    def simulate(self, agents: List[PolicyAgent], steps: int = 10) -> Dict[str, Any]:
        from .social_network import SocialNetwork

        runtime_agents = [agent.to_runtime_dict() for agent in agents]
        self.forum_service.reset()

        network = SocialNetwork(runtime_agents, forum_service=self.forum_service)
        propagation_result = network.simulate_propagation(runtime_agents, steps=steps)

        for agent, runtime_agent in zip(agents, runtime_agents):
            agent.sync_from_runtime(runtime_agent)

        return propagation_result
