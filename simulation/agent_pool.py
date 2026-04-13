"""
智能体池管理

负责基于代表性 roster 与显式政策场景初始化小规模、高信息密度的智能体。
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent import BaseAgent
from util.RoleProfileDB import (
    ROLE_PROFILE_DB_PATH,
    build_profile_org_hints,
    get_all_profiles_for_role,
    init_role_profiles_db,
)
from .models import AgentSpec, PolicyAgent, PolicyScene
from .policy_config import get_policy_simulation_config


class AgentPool:
    """轻量化政策模拟智能体池。"""

    ROLE_COLORS = {
        "accountant": "#3b82f6",
        "auditor": "#10b981",
        "manager": "#8b5cf6",
        "regulator": "#f59e0b",
        "investor": "#ef4444",
    }

    ROLE_NAMES = {
        "accountant": "会计师",
        "auditor": "审计师",
        "manager": "管理层",
        "regulator": "监管者",
        "investor": "投资者",
    }

    ROLE_ORDER = ["accountant", "auditor", "manager", "regulator", "investor"]

    LEVEL_SCORE = {
        "low": 0.25,
        "medium": 0.55,
        "high": 0.85,
        "small": 0.30,
        "large": 0.80,
        "short": 0.75,
        "medium_term": 0.55,
        "long": 0.30,
    }

    def __init__(self, model_type: str = "ollama"):
        self.agents: List[PolicyAgent] = []
        self.model_type = model_type
        self.policy_config = get_policy_simulation_config()
        self.init_config = self.policy_config.get("initialization", {})
        self.role_social_traits = self.policy_config.get(
            "agent_traits", {}
        ).get("role_social_traits", {})
        self.templates = self._load_templates()
        self.template_index = {
            role: {template.get("type"): template for template in role_templates}
            for role, role_templates in self.templates.items()
        }
        self.scene = PolicyScene()
        self.scene_summary: Dict[str, Any] = {}
        self.init_diagnostics: Dict[str, Any] = {}
        self.roster_specs: List[AgentSpec] = []

        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ROLE_PROFILE_DB_PATH,
        )
        if not os.path.exists(db_path):
            init_role_profiles_db(db_path)
        self.role_profile_db_path = db_path

    def initialize(
        self,
        agent_counts: Optional[Dict[str, int]] = None,
        roster_mode: Optional[str] = None,
        scene: Optional[Dict[str, Any]] = None,
        agent_specs: Optional[List[Dict[str, Any]]] = None,
        policy_content: str = "",
    ) -> "AgentPool":
        self.agents = []
        self.scene = self._build_policy_scene(scene=scene, policy_content=policy_content)
        self.roster_specs = self._build_roster(
            agent_counts=agent_counts or {},
            roster_mode=roster_mode,
            agent_specs=agent_specs or [],
        )

        for index, spec in enumerate(self.roster_specs, start=1):
            if spec.role == "investor":
                agent = self._create_investor_agent(spec, index, self.scene)
            else:
                agent = self._create_role_agent(spec, index, self.scene)
            self.agents.append(agent)

        self.scene_summary = self.scene.to_dict()
        self.scene_summary["total_agents"] = len(self.agents)
        self.init_diagnostics = self._build_init_diagnostics()
        return self

    def _load_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        templates: Dict[str, List[Dict[str, Any]]] = {
            "accountant": [],
            "auditor": [],
            "manager": [],
            "regulator": [],
        }
        template_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "simulation",
            "role_templates",
        )

        for role in templates:
            template_dir = os.path.join(template_base_dir, role)
            if not os.path.exists(template_dir):
                continue
            for filename in sorted(os.listdir(template_dir)):
                if not filename.endswith((".yaml", ".yml")):
                    continue
                filepath = os.path.join(template_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as handle:
                        template_data = yaml.safe_load(handle) or {}
                except Exception as exc:
                    print(f"[警告] 加载模板 {filepath} 失败: {exc}")
                    continue
                if isinstance(template_data, dict) and template_data:
                    templates[role].append(template_data)

        if not any(templates.values()):
            templates = self._get_default_templates()
        return templates

    def _get_default_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "accountant": [
                {
                    "type": "conservative",
                    "name": "保守型会计师",
                    "system_prompt": "你是一位资深会计师，优先考虑合规风险和口径一致性。",
                    "concerns": ["合规风险", "披露一致性", "审计风险"],
                    "characteristics": {
                        "policy_sensitivity": 0.80,
                        "conservative_bias": 0.80,
                        "expert_trust": 0.65,
                    },
                },
                {
                    "type": "detail_focused",
                    "name": "细节型会计师",
                    "system_prompt": "你是一位注重准则细节的会计师，逐条核对口径和依据。",
                    "concerns": ["条款细节", "应用案例", "监管解释"],
                    "characteristics": {
                        "policy_sensitivity": 0.72,
                        "conservative_bias": 0.62,
                        "expert_trust": 0.72,
                    },
                },
                {
                    "type": "efficiency_focused",
                    "name": "效率型会计师",
                    "system_prompt": "你是一位强调执行效率的会计师，在合规前提下追求落地速度。",
                    "concerns": ["处理效率", "流程简化", "成本控制"],
                    "characteristics": {
                        "policy_sensitivity": 0.52,
                        "conservative_bias": 0.32,
                        "expert_trust": 0.52,
                    },
                },
            ],
            "auditor": [
                {
                    "type": "strict",
                    "name": "严格型审计师",
                    "system_prompt": "你是一位严格型审计师，优先保证证据充分和程序站得住脚。",
                    "concerns": ["审计风险", "证据充分性", "合规检查"],
                    "characteristics": {
                        "policy_sensitivity": 0.90,
                        "conservative_bias": 0.88,
                        "expert_trust": 0.78,
                    },
                },
                {
                    "type": "risk_auditor",
                    "name": "风险评估型审计师",
                    "system_prompt": "你是一位风险评估型审计师，重视风险预警和识别。",
                    "concerns": ["风险识别", "风险评估", "预防措施"],
                    "characteristics": {
                        "policy_sensitivity": 0.72,
                        "conservative_bias": 0.62,
                        "expert_trust": 0.64,
                    },
                },
                {
                    "type": "pragmatic",
                    "name": "务实型审计师",
                    "system_prompt": "你是一位务实型审计师，在审计质量与效率之间取平衡。",
                    "concerns": ["审计效率", "风险控制", "沟通协调"],
                    "characteristics": {
                        "policy_sensitivity": 0.54,
                        "conservative_bias": 0.52,
                        "expert_trust": 0.52,
                    },
                },
            ],
            "manager": [
                {
                    "type": "profit_focused",
                    "name": "利润导向型管理层",
                    "system_prompt": "你是一位关注业绩和投资者反应的管理者。",
                    "concerns": ["业绩表现", "投资者关系", "融资影响"],
                    "characteristics": {
                        "policy_sensitivity": 0.62,
                        "conservative_bias": 0.32,
                        "expert_trust": 0.42,
                    },
                },
                {
                    "type": "strategic",
                    "name": "战略型管理层",
                    "system_prompt": "你是一位从长期战略视角看待政策影响的管理者。",
                    "concerns": ["战略影响", "长期竞争优势", "资本市场反应"],
                    "characteristics": {
                        "policy_sensitivity": 0.54,
                        "conservative_bias": 0.42,
                        "expert_trust": 0.52,
                    },
                },
                {
                    "type": "risk_averse",
                    "name": "风险规避型管理层",
                    "system_prompt": "你是一位重视合规和声誉风险的管理者。",
                    "concerns": ["合规风险", "声誉管理", "长期发展"],
                    "characteristics": {
                        "policy_sensitivity": 0.70,
                        "conservative_bias": 0.72,
                        "expert_trust": 0.60,
                    },
                },
            ],
            "regulator": [
                {
                    "type": "enforcer",
                    "name": "执行导向型监管者",
                    "system_prompt": "你是一位执行导向型监管者，关注政策落地与市场秩序。",
                    "concerns": ["政策执行", "市场秩序", "违规查处"],
                    "characteristics": {
                        "policy_sensitivity": 0.90,
                        "conservative_bias": 0.42,
                        "expert_trust": 0.72,
                    },
                },
                {
                    "type": "guide",
                    "name": "指导导向型监管者",
                    "system_prompt": "你是一位指导导向型监管者，重视解释与教育。",
                    "concerns": ["政策解读", "市场教育", "反馈收集"],
                    "characteristics": {
                        "policy_sensitivity": 0.80,
                        "conservative_bias": 0.52,
                        "expert_trust": 0.80,
                    },
                },
                {
                    "type": "compliance_focused",
                    "name": "合规导向型监管者",
                    "system_prompt": "你是一位合规导向型监管者，重视执行标准和检查闭环。",
                    "concerns": ["合规执行", "政策落实", "违规查处"],
                    "characteristics": {
                        "policy_sensitivity": 0.88,
                        "conservative_bias": 0.62,
                        "expert_trust": 0.72,
                    },
                },
            ],
        }

    def _build_roster(
        self,
        agent_counts: Dict[str, int],
        roster_mode: Optional[str],
        agent_specs: List[Dict[str, Any]],
    ) -> List[AgentSpec]:
        if agent_specs:
            return [self._normalize_agent_spec(item) for item in agent_specs]
        if any(int(agent_counts.get(role, 0) or 0) > 0 for role in self.ROLE_ORDER):
            return self._build_roster_from_counts(agent_counts)
        return self._build_default_roster(roster_mode)

    def _build_default_roster(self, roster_mode: Optional[str]) -> List[AgentSpec]:
        roster_name = (
            roster_mode
            or self.init_config.get("default_roster_mode")
            or "default"
        )
        raw_roster = self.init_config.get("rosters", {}).get(
            roster_name,
            self.init_config.get("rosters", {}).get("default", []),
        )
        return [self._normalize_agent_spec(item) for item in raw_roster]

    def _build_roster_from_counts(
        self, agent_counts: Dict[str, int]
    ) -> List[AgentSpec]:
        template_order = self.init_config.get("template_order", {})
        roster: List[AgentSpec] = []

        for role in self.ROLE_ORDER:
            count = max(0, int(agent_counts.get(role, 0) or 0))
            order = list(template_order.get(role, []))
            if not order:
                if role == "investor":
                    order = ["value_long_term", "sentiment_trader"]
                else:
                    order = list(self.template_index.get(role, {}).keys())
            if not order:
                continue
            for index in range(count):
                template_type = order[index % len(order)]
                roster.append(
                    AgentSpec(
                        role=role,
                        template_type=template_type,
                        prototype_id=template_type if role == "investor" else None,
                        label=f"{role}_{index + 1}",
                    )
                )

        return roster or self._build_default_roster(None)

    def _normalize_agent_spec(self, raw_spec: Dict[str, Any]) -> AgentSpec:
        spec = AgentSpec.from_dict(raw_spec)
        if spec.role == "investor" and not spec.prototype_id:
            spec.prototype_id = spec.template_type
        return spec

    def _build_policy_scene(
        self, scene: Optional[Dict[str, Any]], policy_content: str = ""
    ) -> PolicyScene:
        scene_inference = self.init_config.get("scene_inference", {})
        default_domain = scene_inference.get("default_domain", "general_compliance")
        default_payload = {
            "policy_domain": default_domain,
            "affected_roles": scene_inference.get("domain_roles", {}).get(
                default_domain, []
            ),
            "affected_industries": scene_inference.get("domain_industries", {}).get(
                default_domain, ["全行业"]
            ),
            "enforcement_intensity": 0.50,
            "ambiguity_level": 0.48,
            "public_attention": 0.50,
            "implementation_urgency": 0.50,
            "disclosure_pressure": 0.52,
            "source": "default",
            "matched_keywords": [],
        }
        inferred_payload = (
            self._infer_scene_from_policy(policy_content) if policy_content else {}
        )
        explicit_payload = dict(scene or {})

        payload = dict(default_payload)
        payload.update(inferred_payload)
        payload.update(explicit_payload)

        if explicit_payload:
            payload["source"] = "user_scene"
        elif inferred_payload:
            payload["source"] = "policy_inference"

        payload["affected_roles"] = self._unique_list(
            payload.get("affected_roles") or default_payload["affected_roles"]
        )
        payload["affected_industries"] = self._unique_list(
            payload.get("affected_industries") or default_payload["affected_industries"]
        )
        payload["matched_keywords"] = self._unique_list(
            payload.get("matched_keywords") or []
        )

        for field in [
            "enforcement_intensity",
            "ambiguity_level",
            "public_attention",
            "implementation_urgency",
            "disclosure_pressure",
        ]:
            payload[field] = round(
                self._clamp(self._safe_float(payload.get(field), 0.5), 0.0, 1.0), 3
            )

        return PolicyScene.from_dict(payload)

    def _infer_scene_from_policy(self, policy_content: str) -> Dict[str, Any]:
        scene_inference = self.init_config.get("scene_inference", {})
        domain_keywords = scene_inference.get("domain_keywords", {})
        domain_roles = scene_inference.get("domain_roles", {})
        domain_industries = scene_inference.get("domain_industries", {})
        industry_keywords = scene_inference.get("industry_keywords", {})
        text = str(policy_content or "")

        domain_scores = {
            domain: self._keyword_hits(text, keywords)
            for domain, keywords in domain_keywords.items()
        }
        default_domain = scene_inference.get("default_domain", "general_compliance")
        selected_domain = max(domain_scores, key=domain_scores.get) if domain_scores else default_domain
        if not domain_scores or domain_scores.get(selected_domain, 0) <= 0:
            selected_domain = default_domain

        matched_keywords = [
            keyword
            for keyword in domain_keywords.get(selected_domain, [])
            if keyword in text
        ][:6]

        affected_industries = []
        for industry, keywords in industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                affected_industries.append(industry)
        if not affected_industries:
            affected_industries = list(domain_industries.get(selected_domain, ["全行业"]))

        enforcement_hits = self._keyword_hits(
            text, ["处罚", "执法", "问询", "检查", "整改", "严查"]
        )
        ambiguity_hits = self._keyword_hits(
            text, ["征求意见", "试行", "探索", "拟", "试点"]
        )
        attention_hits = self._keyword_hits(
            text, ["市场", "投资者", "上市公司", "舆论", "社会", "资本市场"]
        )
        urgency_hits = self._keyword_hits(
            text, ["立即", "尽快", "即日起", "实施", "落地", "执行"]
        )
        disclosure_hits = self._keyword_hits(
            text, ["披露", "公告", "说明", "透明", "财报", "口径"]
        )

        return {
            "policy_domain": selected_domain,
            "affected_roles": domain_roles.get(selected_domain, []),
            "affected_industries": affected_industries,
            "enforcement_intensity": self._bounded_from_hits(0.42, enforcement_hits, 0.12),
            "ambiguity_level": self._bounded_from_hits(0.28, ambiguity_hits, 0.12),
            "public_attention": self._bounded_from_hits(0.38, attention_hits, 0.10),
            "implementation_urgency": self._bounded_from_hits(0.36, urgency_hits, 0.12),
            "disclosure_pressure": self._bounded_from_hits(0.38, disclosure_hits, 0.11),
            "matched_keywords": matched_keywords,
        }

    def _create_role_agent(
        self, spec: AgentSpec, agent_index: int, scene: PolicyScene
    ) -> PolicyAgent:
        template = self._resolve_template(spec.role, spec.template_type)
        profile = self._select_role_profile(spec.role, spec.template_type, agent_index)
        org_context = self._resolve_org_context(spec.role, spec.template_type, profile)
        relevance, relevance_type = self._calculate_policy_relevance(
            spec.role, template, profile, scene
        )
        (
            influence_tier,
            influence_score,
            cross_role_reach,
            cross_role_score,
        ) = self._resolve_influence_tier(spec.role, profile, org_context)
        activation_tier, activation_score, is_active = self._resolve_activation_tier(
            spec.role,
            scene,
            relevance,
            influence_score,
            org_context,
        )
        prior_belief, belief_components = self._calculate_prior_belief(
            spec.role,
            spec.template_type,
            profile,
            org_context,
            scene,
            relevance,
        )

        characteristics = dict(template.get("characteristics", {}) or {})
        voice_style = self._resolve_voice_style(spec.role, spec.template_type)
        social_traits = self._build_social_traits(
            role=spec.role,
            profile=profile,
            characteristics=characteristics,
            relevance=relevance,
            influence_tier=influence_tier,
            activation_tier=activation_tier,
            cross_role_reach=cross_role_reach,
            org_context=org_context,
            scene=scene,
        )
        characteristics.update(
            {
                "peer_influence": social_traits["peer_influence"],
                "influence_score": influence_score,
                "cross_role_reach_score": cross_role_score,
            }
        )
        concerns = self._build_role_concerns(template, profile)
        scenario_context = self._build_scenario_context(
            scene, relevance, relevance_type, activation_tier
        )
        init_trace = self._build_init_trace(
            spec=spec,
            scene=scene,
            relevance=relevance,
            relevance_type=relevance_type,
            prior_belief=prior_belief,
            belief_components=belief_components,
            influence_tier=influence_tier,
            influence_score=influence_score,
            activation_tier=activation_tier,
            activation_score=activation_score,
            cross_role_reach=cross_role_reach,
            cross_role_score=cross_role_score,
            org_context=org_context,
        )

        enhanced_prompt = self._enhance_role_prompt(
            template.get("system_prompt", ""),
            profile,
            org_context,
        )
        llm_agent = None
        try:
            llm_agent = self._build_llm_agent(enhanced_prompt)
        except Exception as exc:
            if self.model_type == "openai":
                raise RuntimeError(f"{self.ROLE_NAMES[spec.role]} 初始化失败: {exc}") from exc
            print(f"[警告] 创建LLM实例失败: {exc}")

        return PolicyAgent(
            id=f"{spec.role}_{agent_index:03d}",
            role=spec.role,
            role_name=self.ROLE_NAMES[spec.role],
            color=self.ROLE_COLORS[spec.role],
            agent_type=spec.template_type,
            template=template,
            system_prompt=enhanced_prompt,
            profile=profile,
            concerns=concerns,
            characteristics=characteristics,
            llm_agent=llm_agent,
            belief=prior_belief,
            initial_belief=prior_belief,
            prior_belief=prior_belief,
            stubbornness=social_traits["stubbornness"],
            activity_level=social_traits["activity_level"],
            susceptibility=social_traits["susceptibility"],
            expression_threshold=social_traits["expression_threshold"],
            repost_tendency=social_traits["repost_tendency"],
            policy_relevance=relevance,
            relevance_type=relevance_type,
            affected_domain=scene.policy_domain,
            enforcement_pressure=scene.enforcement_intensity,
            is_active=is_active,
            activation_tier=activation_tier,
            influence_tier=influence_tier,
            cross_role_reach=cross_role_reach,
            voice_style=voice_style,
            org_context=org_context,
            scenario_context=scenario_context,
            init_trace=init_trace,
        )

    def _create_investor_agent(
        self, spec: AgentSpec, agent_index: int, scene: PolicyScene
    ) -> PolicyAgent:
        profile = self._build_investor_profile(spec, agent_index)
        org_context = self._resolve_org_context("investor", spec.template_type, profile)
        characteristics = self._build_investor_characteristics(profile)
        relevance, relevance_type = self._calculate_policy_relevance(
            "investor", {}, profile, scene
        )
        (
            influence_tier,
            influence_score,
            cross_role_reach,
            cross_role_score,
        ) = self._resolve_influence_tier("investor", profile, org_context)
        activation_tier, activation_score, is_active = self._resolve_activation_tier(
            "investor",
            scene,
            relevance,
            influence_score,
            org_context,
        )
        prior_belief, belief_components = self._calculate_prior_belief(
            "investor",
            spec.template_type,
            profile,
            org_context,
            scene,
            relevance,
        )
        social_traits = self._build_social_traits(
            role="investor",
            profile=profile,
            characteristics=characteristics,
            relevance=relevance,
            influence_tier=influence_tier,
            activation_tier=activation_tier,
            cross_role_reach=cross_role_reach,
            org_context=org_context,
            scene=scene,
        )
        characteristics.update(
            {
                "peer_influence": social_traits["peer_influence"],
                "influence_score": influence_score,
                "cross_role_reach_score": cross_role_score,
            }
        )
        scenario_context = self._build_scenario_context(
            scene, relevance, relevance_type, activation_tier
        )
        init_trace = self._build_init_trace(
            spec=spec,
            scene=scene,
            relevance=relevance,
            relevance_type=relevance_type,
            prior_belief=prior_belief,
            belief_components=belief_components,
            influence_tier=influence_tier,
            influence_score=influence_score,
            activation_tier=activation_tier,
            activation_score=activation_score,
            cross_role_reach=cross_role_reach,
            cross_role_score=cross_role_score,
            org_context=org_context,
        )
        system_prompt = self._build_investor_prompt(profile, org_context)
        llm_agent = None
        try:
            llm_agent = self._build_llm_agent(system_prompt)
        except Exception as exc:
            if self.model_type == "openai":
                raise RuntimeError(f"投资者初始化失败: {exc}") from exc
            print(f"[警告] 创建投资者LLM实例失败: {exc}")

        return PolicyAgent(
            id=f"investor_{agent_index:03d}",
            role="investor",
            role_name=self.ROLE_NAMES["investor"],
            color=self.ROLE_COLORS["investor"],
            agent_type=spec.template_type,
            profile=profile,
            concerns=self._build_investor_concerns(profile, scene),
            characteristics=characteristics,
            llm_agent=llm_agent,
            system_prompt=system_prompt,
            belief=prior_belief,
            initial_belief=prior_belief,
            prior_belief=prior_belief,
            stubbornness=social_traits["stubbornness"],
            activity_level=social_traits["activity_level"],
            susceptibility=social_traits["susceptibility"],
            expression_threshold=social_traits["expression_threshold"],
            repost_tendency=social_traits["repost_tendency"],
            policy_relevance=relevance,
            relevance_type=relevance_type,
            affected_domain=scene.policy_domain,
            enforcement_pressure=scene.enforcement_intensity,
            is_active=is_active,
            activation_tier=activation_tier,
            influence_tier=influence_tier,
            cross_role_reach=cross_role_reach,
            voice_style=self._resolve_voice_style("investor", spec.template_type),
            org_context=org_context,
            scenario_context=scenario_context,
            init_trace=init_trace,
            user_id=profile["user_id"],
        )

    def _resolve_template(self, role: str, template_type: str) -> Dict[str, Any]:
        template = self.template_index.get(role, {}).get(template_type)
        if template:
            return dict(template)
        role_templates = self.templates.get(role, [])
        return dict(role_templates[0]) if role_templates else {}

    def _select_role_profile(
        self, role: str, template_type: str, agent_index: int
    ) -> Dict[str, Any]:
        profiles = get_all_profiles_for_role(role, self.role_profile_db_path)
        typed_profiles = [item for item in profiles if item.get("type") == template_type]
        pool = typed_profiles or profiles
        if not pool:
            return {}

        seed = f"{role}|{template_type}|{agent_index}"
        digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(pool)
        return dict(pool[index])

    def _build_investor_profile(
        self, spec: AgentSpec, agent_index: int
    ) -> Dict[str, Any]:
        prototypes = self.init_config.get("investor_prototypes", {})
        prototype_id = spec.prototype_id or spec.template_type or "value_long_term"
        base_profile = dict(
            prototypes.get(prototype_id)
            or next(iter(prototypes.values()), {})
        )
        base_profile.setdefault("followed_industries", ["资本市场"])
        base_profile["user_id"] = f"{prototype_id}_{agent_index:03d}"
        base_profile.setdefault("name", base_profile.get("self_description", "投资者"))
        return base_profile

    def _resolve_org_context(
        self, role: str, template_type: str, profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        if role == "investor":
            return self._resolve_investor_org_context(profile)

        hints = build_profile_org_hints(role, template_type, profile)
        context = {
            "organization_type": hints.get(
                "organization_type",
                profile.get("organization_type", "相关机构"),
            ),
            "organization_size": hints.get("organization_size", "medium"),
            "regulatory_exposure": hints.get("regulatory_exposure", "medium"),
            "decision_authority": hints.get("decision_authority", "medium"),
            "accountability_pressure": hints.get("accountability_pressure", "medium"),
            "resource_constraint": hints.get("resource_constraint", "medium"),
        }
        context["organization_size_score"] = self._level_to_score(
            context["organization_size"]
        )
        context["regulatory_exposure_score"] = self._level_to_score(
            context["regulatory_exposure"]
        )
        context["decision_authority_score"] = self._level_to_score(
            context["decision_authority"]
        )
        context["accountability_pressure_score"] = self._level_to_score(
            context["accountability_pressure"]
        )
        context["resource_constraint_score"] = self._level_to_score(
            context["resource_constraint"]
        )
        return context

    def _resolve_investor_org_context(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        capital_type = str(profile.get("capital_type", "thematic_account"))
        holding_horizon = str(profile.get("holding_horizon", "medium"))
        risk_budget = str(profile.get("risk_budget", "medium"))
        information_sensitivity = str(
            profile.get("information_sensitivity", "medium")
        )

        if capital_type == "institutional_long_only":
            organization_size = "large"
            decision_authority = "high"
            accountability_pressure = "high"
            regulatory_exposure = "medium"
            resource_constraint = "low"
        elif capital_type == "fast_money":
            organization_size = "medium"
            decision_authority = "medium"
            accountability_pressure = "medium"
            regulatory_exposure = "low"
            resource_constraint = "low"
        elif capital_type == "event_fund":
            organization_size = "medium"
            decision_authority = "high"
            accountability_pressure = "medium"
            regulatory_exposure = "medium"
            resource_constraint = "medium"
        else:
            organization_size = "medium"
            decision_authority = "medium"
            accountability_pressure = "medium"
            regulatory_exposure = "medium"
            resource_constraint = "medium"

        context = {
            "organization_type": profile.get("organization_type", "投资机构"),
            "organization_size": organization_size,
            "regulatory_exposure": regulatory_exposure,
            "decision_authority": decision_authority,
            "accountability_pressure": accountability_pressure,
            "resource_constraint": resource_constraint,
            "capital_type": capital_type,
            "holding_horizon": holding_horizon,
            "risk_budget": risk_budget,
            "information_sensitivity": information_sensitivity,
        }
        context["organization_size_score"] = self._level_to_score(
            context["organization_size"]
        )
        context["regulatory_exposure_score"] = self._level_to_score(
            context["regulatory_exposure"]
        )
        context["decision_authority_score"] = self._level_to_score(
            context["decision_authority"]
        )
        context["accountability_pressure_score"] = self._level_to_score(
            context["accountability_pressure"]
        )
        context["resource_constraint_score"] = self._level_to_score(
            context["resource_constraint"]
        )
        context["risk_budget_score"] = self._level_to_score(risk_budget)
        context["information_sensitivity_score"] = self._level_to_score(
            information_sensitivity
        )
        context["holding_horizon_score"] = self._level_to_score(
            "medium_term" if holding_horizon == "medium" else holding_horizon
        )
        return context

    def _calculate_policy_relevance(
        self,
        role: str,
        template: Dict[str, Any],
        profile: Dict[str, Any],
        scene: PolicyScene,
    ) -> Tuple[float, str]:
        weights = self.init_config.get("relevance_weights", {})
        affinity = self.init_config.get("role_domain_affinity", {})
        domain_keywords = self.init_config.get("scene_inference", {}).get(
            "domain_keywords", {}
        )

        score = self._safe_float(
            affinity.get(scene.policy_domain, {}).get(role),
            0.45,
        )
        if role in scene.affected_roles:
            score += self._safe_float(weights.get("direct_role_boost"), 0.15)

        industries = self._profile_industry_tokens(profile)
        if scene.affected_industries and "全行业" not in scene.affected_industries:
            if any(industry in industries for industry in scene.affected_industries):
                score += self._safe_float(weights.get("industry_match_boost"), 0.10)

        domain_terms = scene.matched_keywords or domain_keywords.get(scene.policy_domain, [])
        specialization_text = " ".join(
            [
                str(profile.get("specialization", "")),
                str(profile.get("organization_type", "")),
                str(template.get("name", "")),
            ]
        )
        if any(term and term in specialization_text for term in domain_terms):
            score += self._safe_float(weights.get("specialization_match_boost"), 0.08)

        if role in {"accountant", "auditor", "regulator"}:
            score += scene.enforcement_intensity * self._safe_float(
                weights.get("enforcement_scale"), 0.10
            )
        if role in {"manager", "investor"}:
            score += scene.public_attention * self._safe_float(
                weights.get("public_attention_scale"), 0.06
            )
        if role == "investor":
            score += scene.disclosure_pressure * 0.08
        if role == "manager":
            score += scene.implementation_urgency * 0.06

        score = round(self._clamp(score, 0.0, 1.0), 3)
        return score, self._relevance_type(score)

    def _calculate_prior_belief(
        self,
        role: str,
        template_type: str,
        profile: Dict[str, Any],
        org_context: Dict[str, Any],
        scene: PolicyScene,
        relevance: float,
    ) -> Tuple[float, Dict[str, float]]:
        weights = self.init_config.get("prior_belief_weights", {})
        role_base = self._safe_float(
            self.init_config.get("role_base_bias", {}).get(role),
            0.0,
        )
        template_bias = self._safe_float(
            self.init_config.get("template_bias", {}).get(role, {}).get(template_type),
            0.0,
        )
        direction_seed = role_base + template_bias
        if abs(direction_seed) < 0.05:
            direction_seed = self._safe_float(
                self.init_config.get("role_direction", {}).get(role),
                0.0,
            )
        direction = 0.0 if abs(direction_seed) < 0.01 else (1.0 if direction_seed > 0 else -1.0)

        authority_score = self._safe_float(
            org_context.get("decision_authority_score"), 0.55
        )
        accountability_score = self._safe_float(
            org_context.get("accountability_pressure_score"), 0.55
        )
        exposure_score = self._safe_float(
            org_context.get("regulatory_exposure_score"), 0.55
        )
        resource_score = self._safe_float(
            org_context.get("resource_constraint_score"), 0.55
        )

        org_pressure = (
            (authority_score - 0.5)
            * self._safe_float(weights.get("authority_bonus"), 0.12)
            - (accountability_score - 0.5)
            * self._safe_float(weights.get("accountability_penalty"), 0.18)
            - (exposure_score - 0.5)
            * self._safe_float(weights.get("exposure_penalty"), 0.12)
            - (resource_score - 0.5)
            * self._safe_float(weights.get("resource_penalty"), 0.08)
        )

        if role == "regulator":
            org_pressure += (exposure_score - 0.5) * 0.18 + (authority_score - 0.5) * 0.06
        elif role == "manager":
            org_pressure += (authority_score - 0.5) * 0.08
        elif role == "investor":
            risk_budget_score = self._safe_float(org_context.get("risk_budget_score"), 0.55)
            info_score = self._safe_float(
                org_context.get("information_sensitivity_score"), 0.55
            )
            horizon = str(org_context.get("holding_horizon", "medium"))
            org_pressure += (
                (risk_budget_score - 0.5)
                * self._safe_float(weights.get("investor_risk_budget_bonus"), 0.16)
                + (info_score - 0.5) * 0.08
            )
            if horizon == "long":
                org_pressure -= 0.05
            elif horizon == "short":
                org_pressure += 0.06

        if direction == 0.0:
            relevance_bias = (relevance - 0.5) * 0.12
        else:
            relevance_bias = (
                relevance
                * self._safe_float(weights.get("relevance_scale"), 0.28)
                * direction
            )

        noise = (
            self._stable_noise(role, template_type, profile.get("name") or profile.get("user_id"))
            * self._safe_float(weights.get("noise_scale"), 0.15)
        )

        raw_belief = role_base + template_bias + org_pressure + relevance_bias + noise
        ambiguity_factor = 1 - scene.ambiguity_level * self._safe_float(
            weights.get("ambiguity_dampening"), 0.16
        )
        belief = round(self._clamp(raw_belief * ambiguity_factor, -1.0, 1.0), 3)

        return belief, {
            "role_base": round(role_base, 3),
            "template_bias": round(template_bias, 3),
            "org_pressure": round(org_pressure, 3),
            "policy_relevance_bias": round(relevance_bias, 3),
            "stable_noise": round(noise, 3),
            "ambiguity_factor": round(ambiguity_factor, 3),
        }

    def _resolve_influence_tier(
        self, role: str, profile: Dict[str, Any], org_context: Dict[str, Any]
    ) -> Tuple[str, float, str, float]:
        thresholds = self.init_config.get("tier_thresholds", {})
        network_influence = self._safe_float(profile.get("network_influence"), 0.5)
        reputation_score = self._safe_float(profile.get("reputation_score"), 0.5)
        authority_score = self._safe_float(
            org_context.get("decision_authority_score"), 0.55
        )
        org_size_score = self._safe_float(
            org_context.get("organization_size_score"), 0.55
        )

        influence_score = (
            0.32 * network_influence
            + 0.28 * reputation_score
            + 0.25 * authority_score
            + 0.15 * org_size_score
        )
        if role == "regulator":
            influence_score += 0.08
        elif role == "manager":
            influence_score += 0.04

        influence_score = round(self._clamp(influence_score, 0.0, 1.0), 3)
        influence_tier = self._tier_from_score(
            influence_score,
            self._safe_float(thresholds.get("influence", {}).get("high"), 0.72),
            self._safe_float(thresholds.get("influence", {}).get("medium"), 0.50),
        )

        cross_role_score = (
            0.40 * network_influence
            + 0.20 * reputation_score
            + 0.20 * org_size_score
            + 0.20 * authority_score
        )
        if role in {"manager", "regulator", "investor"}:
            cross_role_score += 0.06
        cross_role_score = round(self._clamp(cross_role_score, 0.0, 1.0), 3)
        cross_role_tier = self._tier_from_score(
            cross_role_score,
            self._safe_float(thresholds.get("cross_role", {}).get("high"), 0.72),
            self._safe_float(thresholds.get("cross_role", {}).get("medium"), 0.45),
        )

        return influence_tier, influence_score, cross_role_tier, cross_role_score

    def _resolve_activation_tier(
        self,
        role: str,
        scene: PolicyScene,
        relevance: float,
        influence_score: float,
        org_context: Dict[str, Any],
    ) -> Tuple[str, float, bool]:
        thresholds = self.init_config.get("tier_thresholds", {}).get("activation", {})
        authority_score = self._safe_float(
            org_context.get("decision_authority_score"), 0.55
        )
        resource_score = self._safe_float(
            org_context.get("resource_constraint_score"), 0.55
        )
        activation_score = (
            0.58 * relevance
            + 0.16 * scene.public_attention
            + 0.14 * influence_score
            + 0.12 * authority_score
            - 0.08 * resource_score
        )
        if role in scene.affected_roles:
            activation_score += 0.06
        if role == "regulator":
            activation_score += scene.enforcement_intensity * 0.06
        elif role == "investor":
            activation_score += scene.public_attention * 0.05

        activation_score = round(self._clamp(activation_score, 0.0, 1.0), 3)
        if activation_score >= self._safe_float(thresholds.get("lead"), 0.74):
            activation_tier = "lead"
        elif activation_score <= self._safe_float(thresholds.get("observe"), 0.38):
            activation_tier = "observe"
        else:
            activation_tier = "react"

        is_active = activation_tier != "observe" or influence_score >= 0.78
        return activation_tier, activation_score, is_active

    def _build_social_traits(
        self,
        role: str,
        profile: Optional[Dict[str, Any]] = None,
        characteristics: Optional[Dict[str, Any]] = None,
        relevance: float = 0.5,
        influence_tier: str = "medium",
        activation_tier: str = "react",
        cross_role_reach: str = "medium",
        org_context: Optional[Dict[str, Any]] = None,
        scene: Optional[PolicyScene] = None,
    ) -> Dict[str, float]:
        default_traits = {
            "stubbornness": 0.5,
            "activity_level": 0.5,
            "susceptibility": 0.5,
            "expression_threshold": 0.2,
            "repost_tendency": 0.2,
        }
        traits = self.role_social_traits.get(role, default_traits).copy()
        profile = profile or {}
        characteristics = characteristics or {}
        scene = scene or PolicyScene()

        network_influence = self._safe_float(profile.get("network_influence"), 0.5)
        reputation_score = self._safe_float(profile.get("reputation_score"), 0.5)
        experience_years = self._safe_float(profile.get("experience_years"), 8.0)
        experience_factor = self._clamp(experience_years / 20.0, 0.2, 1.0)
        policy_sensitivity = self._safe_float(
            characteristics.get("policy_sensitivity"), 0.6
        )
        conservative_bias = self._safe_float(
            characteristics.get("conservative_bias"), 0.5
        )
        expert_trust = self._safe_float(characteristics.get("expert_trust"), 0.5)
        risk_signal = self._resolve_risk_signal(profile.get("risk_attitude"))
        decision_signal = self._resolve_decision_signal(profile.get("decision_style"))
        authority_score = self._safe_float(
            (org_context or {}).get("decision_authority_score"), 0.55
        )
        noise = self._stable_noise(
            role,
            profile.get("name") or profile.get("user_id"),
            profile.get("organization_type"),
        )

        traits["stubbornness"] = round(
            self._clamp(
                traits["stubbornness"]
                + (conservative_bias - 0.5) * 0.22
                + (experience_factor - 0.5) * 0.08
                + (expert_trust - 0.5) * 0.06
                - risk_signal * 0.04
                + noise * 0.05
            ),
            3,
        )

        activation_adjustment = {"lead": 0.18, "react": 0.02, "observe": -0.18}
        influence_adjustment = {"high": 0.14, "medium": 0.04, "low": -0.05}
        cross_role_adjustment = {"high": 0.10, "medium": 0.04, "low": -0.03}

        traits["activity_level"] = round(
            self._clamp(
                traits["activity_level"]
                + (network_influence - 0.5) * 0.16
                + (policy_sensitivity - 0.5) * 0.12
                + activation_adjustment.get(activation_tier, 0.0)
                + influence_adjustment.get(influence_tier, 0.0)
                + (relevance - 0.5) * 0.20
                + scene.public_attention * 0.05
                + decision_signal * 0.04
                + noise * 0.04
            ),
            3,
        )
        traits["susceptibility"] = round(
            self._clamp(
                traits["susceptibility"]
                - (traits["stubbornness"] - 0.5) * 0.28
                + (expert_trust - 0.5) * 0.08
                - (reputation_score - 0.5) * 0.06
                + risk_signal * 0.04
                - authority_score * 0.04
            ),
            3,
        )
        traits["expression_threshold"] = round(
            self._clamp(
                traits["expression_threshold"]
                + (conservative_bias - 0.5) * 0.06
                - (traits["activity_level"] - 0.5) * 0.06
                - influence_adjustment.get(influence_tier, 0.0) * 0.18
                - activation_adjustment.get(activation_tier, 0.0) * 0.15
                + (0.5 - relevance) * 0.08,
                0.06,
                0.38,
            ),
            3,
        )
        traits["repost_tendency"] = round(
            self._clamp(
                traits["repost_tendency"]
                + (network_influence - 0.5) * 0.12
                + cross_role_adjustment.get(cross_role_reach, 0.0)
                + scene.public_attention * 0.08
                + scene.disclosure_pressure * 0.05
                + noise * 0.04
            ),
            3,
        )
        traits["peer_influence"] = round(
            self._clamp(
                0.20
                + network_influence * 0.22
                + reputation_score * 0.18
                + authority_score * 0.16
                + expert_trust * 0.10
                + influence_adjustment.get(influence_tier, 0.0)
                + cross_role_adjustment.get(cross_role_reach, 0.0) * 0.6
                + noise * 0.03
            ),
            3,
        )

        return traits

    def _build_role_concerns(
        self, template: Dict[str, Any], profile: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        concerns = list(template.get("concerns", []))
        profile = profile or {}

        specialization = str(profile.get("specialization", "")).strip()
        industry_focus = str(profile.get("industry_focus", "")).strip()
        organization_type = str(profile.get("organization_type", "")).strip()
        decision_style = str(profile.get("decision_style", "")).strip()

        derived = []
        if specialization:
            derived.append(f"{specialization}执行风险")
        if industry_focus:
            derived.append(f"{industry_focus}行业适配")
        if organization_type:
            derived.append(f"{organization_type}实施约束")
        if decision_style:
            derived.append(f"{decision_style}落地可行性")

        return self._unique_list(concerns + derived)[:6]

    def _build_investor_concerns(
        self, profile: Dict[str, Any], scene: PolicyScene
    ) -> List[str]:
        concerns = ["信息透明度", "估值重估", "仓位调整", "风险暴露"]
        if scene.policy_domain == "disclosure":
            concerns.insert(0, "披露质量")
        elif scene.policy_domain == "valuation":
            concerns.insert(0, "预期差交易")
        elif scene.policy_domain == "enforcement":
            concerns.insert(0, "监管冲击")

        industries = profile.get("followed_industries", []) or []
        concerns.extend(f"{industry}板块影响" for industry in industries[:2])
        return self._unique_list(concerns)[:6]

    def _build_investor_characteristics(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        risk_budget = str(profile.get("risk_budget", "medium"))
        holding_horizon = str(profile.get("holding_horizon", "medium"))
        info_sensitivity = str(profile.get("information_sensitivity", "medium"))

        conservative_bias = {
            "high": 0.28,
            "medium": 0.50,
            "low": 0.68,
        }.get(risk_budget, 0.50)
        if holding_horizon == "long":
            conservative_bias += 0.10
        elif holding_horizon == "short":
            conservative_bias -= 0.08

        policy_sensitivity = {"high": 0.72, "medium": 0.58, "low": 0.46}.get(
            info_sensitivity, 0.58
        )
        expert_trust = 0.52 if holding_horizon == "long" else 0.42

        return {
            "policy_sensitivity": self._clamp(policy_sensitivity, 0.0, 1.0),
            "conservative_bias": self._clamp(conservative_bias, 0.0, 1.0),
            "expert_trust": self._clamp(expert_trust, 0.0, 1.0),
        }

    def _resolve_voice_style(self, role: str, agent_type: str) -> str:
        voice_styles = self.init_config.get("voice_styles", {})
        return str(voice_styles.get(agent_type) or voice_styles.get(role) or "专业克制")

    def _build_scenario_context(
        self,
        scene: PolicyScene,
        relevance: float,
        relevance_type: str,
        activation_tier: str,
    ) -> Dict[str, Any]:
        return {
            "policy_domain": scene.policy_domain,
            "affected_roles": list(scene.affected_roles),
            "affected_industries": list(scene.affected_industries),
            "policy_relevance": relevance,
            "relevance_type": relevance_type,
            "activation_tier": activation_tier,
            "enforcement_intensity": scene.enforcement_intensity,
            "ambiguity_level": scene.ambiguity_level,
            "public_attention": scene.public_attention,
            "implementation_urgency": scene.implementation_urgency,
            "disclosure_pressure": scene.disclosure_pressure,
            "matched_keywords": list(scene.matched_keywords),
            "source": scene.source,
        }

    def _build_init_trace(
        self,
        spec: AgentSpec,
        scene: PolicyScene,
        relevance: float,
        relevance_type: str,
        prior_belief: float,
        belief_components: Dict[str, float],
        influence_tier: str,
        influence_score: float,
        activation_tier: str,
        activation_score: float,
        cross_role_reach: str,
        cross_role_score: float,
        org_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "agent_spec": spec.to_dict(),
            "scene": {
                "policy_domain": scene.policy_domain,
                "source": scene.source,
                "matched_keywords": list(scene.matched_keywords),
            },
            "policy_relevance": {
                "score": relevance,
                "type": relevance_type,
            },
            "prior_belief": {
                "score": prior_belief,
                "components": belief_components,
            },
            "tiers": {
                "influence_tier": influence_tier,
                "influence_score": influence_score,
                "activation_tier": activation_tier,
                "activation_score": activation_score,
                "cross_role_reach": cross_role_reach,
                "cross_role_score": cross_role_score,
            },
            "org_context": {
                key: value
                for key, value in org_context.items()
                if not key.endswith("_score")
            },
        }

    def _build_llm_agent(self, system_prompt: str) -> BaseAgent:
        config_path = None
        if self.model_type != "ollama":
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            openai_config = os.path.join(project_root, "config", "openai.yaml")
            api_config = os.path.join(project_root, "config", "api.yaml")

            if os.path.exists(openai_config):
                config_path = "config/openai.yaml"
            elif os.path.exists(api_config):
                with open(api_config, "r", encoding="utf-8") as handle:
                    api_config_data = yaml.safe_load(handle) or {}

                api_keys = api_config_data.get("api_key") or []
                if api_keys == ["ollama"]:
                    raise FileNotFoundError(
                        "当前 config/api.yaml 仍是 Ollama 配置，无法用于 openai 模式；"
                        "请提供 config/openai.yaml 或将 config/api.yaml 替换为 OpenAI 配置"
                    )
                config_path = "config/api.yaml"
            else:
                raise FileNotFoundError(
                    "缺少 OpenAI 配置文件；请提供 config/openai.yaml 或 config/api.yaml"
                )

        return BaseAgent(system_prompt=system_prompt, config_path=config_path)

    def _enhance_role_prompt(
        self,
        base_prompt: str,
        profile: Dict[str, Any],
        org_context: Dict[str, Any],
    ) -> str:
        if not profile:
            return base_prompt

        past_cases = profile.get("past_cases") or []
        if not isinstance(past_cases, list):
            past_cases = []

        enhancement = f"""

## 个人画像补充
- 工作年限：{profile.get('experience_years', 0)}年
- 服务机构：{profile.get('organization_type', '未知')}
- 专业领域：{profile.get('specialization', '通用')}
- 风险态度：{profile.get('risk_attitude', '中立')}
- 决策风格：{profile.get('decision_style', '标准')}
- 关注行业：{profile.get('industry_focus', '多元化')}
- 组织规模：{org_context.get('organization_size', 'medium')}
- 决策权限：{org_context.get('decision_authority', 'medium')}
- 问责压力：{org_context.get('accountability_pressure', 'medium')}
- 资源约束：{org_context.get('resource_constraint', 'medium')}
- 网络影响力：{profile.get('network_influence', 0.5)}
- 声誉评分：{profile.get('reputation_score', 0.5)}

## 经典案例经历
{chr(10).join(f'- {case}' for case in past_cases[:3]) if past_cases else '- 暂无典型案例'}

请始终保持真实的角色身份，结合你的画像与组织约束进行分析和判断。"""

        return base_prompt + enhancement

    def _build_investor_prompt(
        self, profile: Dict[str, Any], org_context: Dict[str, Any]
    ) -> str:
        industries = "、".join(profile.get("followed_industries", [])[:3]) or "多行业"
        return f"""# 角色定义
你是一位具有明确资金属性和交易边界的投资者。

## 基本信息
- 身份：{profile.get('name', '投资者')}
- 自我描述：{profile.get('self_description', '市场参与者')}
- 投资策略：{profile.get('strategy', '混合')}
- 关注行业：{industries}

## 组织约束
- 机构类型：{profile.get('organization_type', '投资机构')}
- 资本类型：{org_context.get('capital_type', 'thematic_account')}
- 持有周期：{org_context.get('holding_horizon', 'medium')}
- 风险预算：{org_context.get('risk_budget', 'medium')}
- 信息敏感度：{org_context.get('information_sensitivity', 'medium')}

## 决策要求
当面对政策或市场信息时，请从仓位、预期差、估值锚和流动性风险角度进行分析。
不要泛泛而谈，要体现你所代表资金的真实交易边界。"""

    def _resolve_risk_signal(self, risk_attitude: Any) -> float:
        text = str(risk_attitude or "")
        if any(keyword in text for keyword in ["高度规避", "中度规避", "警惕", "稳健"]):
            return -1.0
        if any(keyword in text for keyword in ["冒险", "创新", "主动", "业绩导向"]):
            return 1.0
        return 0.0

    def _resolve_decision_signal(self, decision_style: Any) -> float:
        text = str(decision_style or "")
        if any(keyword in text for keyword in ["严格", "优先", "遵循", "导向", "预期"]):
            return 0.4
        if any(keyword in text for keyword in ["协调", "灵活", "反馈", "教育", "兼顾"]):
            return 0.1
        return 0.0

    def _profile_industry_tokens(self, profile: Dict[str, Any]) -> List[str]:
        tokens = []
        for field in ["industry_focus", "followed_industries"]:
            value = profile.get(field)
            if isinstance(value, list):
                tokens.extend(str(item) for item in value if item)
            elif value:
                tokens.extend(
                    token
                    for token in str(value).replace("、", ",").split(",")
                    if token
                )
        return self._unique_list(tokens)

    def _build_init_diagnostics(self) -> Dict[str, Any]:
        prior_by_role: Dict[str, Dict[str, float]] = {}
        active_counts = {"active": 0, "inactive": 0}
        influence_counts = {"high": 0, "medium": 0, "low": 0}
        relevance_counts = {"high": 0, "medium": 0, "low": 0}
        activation_counts = {"lead": 0, "react": 0, "observe": 0}

        for agent in self.agents:
            role_stats = prior_by_role.setdefault(
                agent.role,
                {"count": 0, "min": 1.0, "max": -1.0, "sum": 0.0},
            )
            role_stats["count"] += 1
            role_stats["sum"] += agent.prior_belief
            role_stats["min"] = min(role_stats["min"], agent.prior_belief)
            role_stats["max"] = max(role_stats["max"], agent.prior_belief)

            if agent.is_active:
                active_counts["active"] += 1
            else:
                active_counts["inactive"] += 1

            influence_counts[agent.influence_tier] = (
                influence_counts.get(agent.influence_tier, 0) + 1
            )
            activation_counts[agent.activation_tier] = (
                activation_counts.get(agent.activation_tier, 0) + 1
            )
            relevance_counts[self._relevance_band(agent.policy_relevance)] += 1

        for stats in prior_by_role.values():
            if stats["count"]:
                stats["avg"] = round(stats["sum"] / stats["count"], 3)
            stats["min"] = round(stats["min"], 3)
            stats["max"] = round(stats["max"], 3)
            stats.pop("sum", None)

        return {
            "prior_belief_by_role": prior_by_role,
            "active_counts": active_counts,
            "influence_tiers": influence_counts,
            "activation_tiers": activation_counts,
            "relevance_bands": relevance_counts,
        }

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for agent in self.agents:
            role_summary = summary.setdefault(
                agent.role,
                {
                    "count": 0,
                    "types": [],
                    "avg_prior_belief": 0.0,
                    "avg_policy_relevance": 0.0,
                    "active_count": 0,
                },
            )
            role_summary["count"] += 1
            role_summary["avg_prior_belief"] += agent.prior_belief
            role_summary["avg_policy_relevance"] += agent.policy_relevance
            if agent.is_active:
                role_summary["active_count"] += 1
            if agent.agent_type:
                role_summary["types"].append(agent.agent_type)

        for role_summary in summary.values():
            if role_summary["count"]:
                role_summary["avg_prior_belief"] = round(
                    role_summary["avg_prior_belief"] / role_summary["count"], 3
                )
                role_summary["avg_policy_relevance"] = round(
                    role_summary["avg_policy_relevance"] / role_summary["count"], 3
                )

        return summary

    def get_scene_summary(self) -> Dict[str, Any]:
        return dict(self.scene_summary)

    def get_init_diagnostics(self) -> Dict[str, Any]:
        return dict(self.init_diagnostics)

    def get_all_profiles(self) -> List[Dict[str, Any]]:
        return [agent.to_public_dict() for agent in self.agents]

    def _relevance_type(self, score: float) -> str:
        if score >= 0.75:
            return "direct"
        if score >= 0.45:
            return "adjacent"
        return "ambient"

    def _relevance_band(self, score: float) -> str:
        if score >= 0.70:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    def _tier_from_score(self, score: float, high_threshold: float, medium_threshold: float) -> str:
        if score >= high_threshold:
            return "high"
        if score >= medium_threshold:
            return "medium"
        return "low"

    def _keyword_hits(self, text: str, keywords: List[str]) -> int:
        return sum(1 for keyword in keywords if keyword and keyword in text)

    def _bounded_from_hits(self, base: float, hits: int, step: float) -> float:
        return round(self._clamp(base + hits * step, 0.0, 1.0), 3)

    def _level_to_score(self, level: Any) -> float:
        return self._safe_float(self.LEVEL_SCORE.get(str(level), 0.55), 0.55)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))

    def _stable_noise(self, *parts: Any) -> float:
        seed = "|".join(str(part or "") for part in parts)
        digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
        return (int(digest[:8], 16) / 0xFFFFFFFF) - 0.5

    def _unique_list(self, values: List[Any]) -> List[Any]:
        deduped = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
