from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _safe_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_list(value: Optional[List[Any]]) -> List[Any]:
    return list(value) if isinstance(value, list) else []


@dataclass
class AgentSpec:
    role: str
    template_type: str
    prototype_id: Optional[str] = None
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentSpec":
        payload = _safe_dict(data)
        return cls(
            role=str(payload.get("role", "")),
            template_type=str(
                payload.get("template_type")
                or payload.get("agent_type")
                or payload.get("prototype_id")
                or ""
            ),
            prototype_id=payload.get("prototype_id"),
            label=str(payload.get("label", "")),
            metadata=_safe_dict(payload.get("metadata")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyScene:
    policy_domain: str = "general_compliance"
    affected_roles: List[str] = field(default_factory=list)
    affected_industries: List[str] = field(default_factory=list)
    enforcement_intensity: float = 0.5
    ambiguity_level: float = 0.5
    public_attention: float = 0.5
    implementation_urgency: float = 0.5
    disclosure_pressure: float = 0.5
    source: str = "default"
    matched_keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PolicyScene":
        payload = _safe_dict(data)
        return cls(
            policy_domain=str(payload.get("policy_domain", "general_compliance")),
            affected_roles=[
                str(item) for item in _safe_list(payload.get("affected_roles"))
            ],
            affected_industries=[
                str(item) for item in _safe_list(payload.get("affected_industries"))
            ],
            enforcement_intensity=float(payload.get("enforcement_intensity", 0.5)),
            ambiguity_level=float(payload.get("ambiguity_level", 0.5)),
            public_attention=float(payload.get("public_attention", 0.5)),
            implementation_urgency=float(payload.get("implementation_urgency", 0.5)),
            disclosure_pressure=float(payload.get("disclosure_pressure", 0.5)),
            source=str(payload.get("source", "default")),
            matched_keywords=[
                str(item) for item in _safe_list(payload.get("matched_keywords"))
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReactionResult:
    understanding: str = "该政策涉及会计处理相关内容"
    decision_stance: str = "中立"
    impact_level: str = "中"
    concerns_analysis: str = "需要进一步分析政策影响"
    recommendation: str = "建议等待更多信息后再决策"
    decision_reason: str = "当前信息有限，需谨慎处理"
    belief_score: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ReactionResult":
        payload = _safe_dict(data)
        return cls(
            understanding=str(payload.get("understanding", cls.understanding)),
            decision_stance=str(payload.get("decision_stance", cls.decision_stance)),
            impact_level=str(payload.get("impact_level", cls.impact_level)),
            concerns_analysis=str(
                payload.get("concerns_analysis", cls.concerns_analysis)
            ),
            recommendation=str(payload.get("recommendation", cls.recommendation)),
            decision_reason=str(payload.get("decision_reason", cls.decision_reason)),
            belief_score=float(payload.get("belief_score", cls.belief_score)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyAgent:
    id: str
    role: str
    role_name: str
    color: str
    agent_type: str = "unknown"
    template: Dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    profile: Dict[str, Any] = field(default_factory=dict)
    concerns: List[str] = field(default_factory=list)
    characteristics: Dict[str, Any] = field(default_factory=dict)
    llm_agent: Any = None
    reaction: Optional[ReactionResult] = None
    belief: float = 0.0
    initial_belief: float = 0.0
    prior_belief: float = 0.0
    stubbornness: float = 0.5
    activity_level: float = 0.5
    susceptibility: float = 0.5
    expression_threshold: float = 0.2
    repost_tendency: float = 0.2
    policy_relevance: float = 0.0
    relevance_type: str = "ambient"
    affected_domain: str = "general_compliance"
    enforcement_pressure: float = 0.5
    is_active: bool = True
    activation_tier: str = "react"
    influence_tier: str = "medium"
    cross_role_reach: str = "medium"
    voice_style: str = "专业克制"
    org_context: Dict[str, Any] = field(default_factory=dict)
    scenario_context: Dict[str, Any] = field(default_factory=dict)
    init_trace: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyAgent":
        payload = _safe_dict(data)
        reaction = payload.get("reaction")
        return cls(
            id=str(payload["id"]),
            role=str(payload["role"]),
            role_name=str(payload["role_name"]),
            color=str(payload["color"]),
            agent_type=str(payload.get("agent_type", "unknown")),
            template=_safe_dict(payload.get("template")),
            system_prompt=str(payload.get("system_prompt", "")),
            profile=_safe_dict(payload.get("profile")),
            concerns=_safe_list(payload.get("concerns")),
            characteristics=_safe_dict(payload.get("characteristics")),
            llm_agent=payload.get("llm_agent"),
            reaction=ReactionResult.from_dict(reaction) if reaction else None,
            belief=float(payload.get("belief", 0.0)),
            initial_belief=float(payload.get("initial_belief", 0.0)),
            prior_belief=float(payload.get("prior_belief", 0.0)),
            stubbornness=float(payload.get("stubbornness", 0.5)),
            activity_level=float(payload.get("activity_level", 0.5)),
            susceptibility=float(payload.get("susceptibility", 0.5)),
            expression_threshold=float(payload.get("expression_threshold", 0.2)),
            repost_tendency=float(payload.get("repost_tendency", 0.2)),
            policy_relevance=float(payload.get("policy_relevance", 0.0)),
            relevance_type=str(payload.get("relevance_type", "ambient")),
            affected_domain=str(
                payload.get("affected_domain", "general_compliance")
            ),
            enforcement_pressure=float(payload.get("enforcement_pressure", 0.5)),
            is_active=bool(payload.get("is_active", True)),
            activation_tier=str(payload.get("activation_tier", "react")),
            influence_tier=str(payload.get("influence_tier", "medium")),
            cross_role_reach=str(payload.get("cross_role_reach", "medium")),
            voice_style=str(payload.get("voice_style", "专业克制")),
            org_context=_safe_dict(payload.get("org_context")),
            scenario_context=_safe_dict(payload.get("scenario_context")),
            init_trace=_safe_dict(payload.get("init_trace")),
            user_id=payload.get("user_id"),
        )

    def apply_reaction(self, reaction: ReactionResult) -> None:
        self.reaction = reaction
        self.belief = float(reaction.belief_score)
        self.initial_belief = float(reaction.belief_score)

    def sync_from_runtime(self, payload: Dict[str, Any]) -> None:
        self.profile = _safe_dict(payload.get("profile"))
        self.concerns = _safe_list(payload.get("concerns"))
        self.characteristics = _safe_dict(payload.get("characteristics"))
        self.belief = float(payload.get("belief", self.belief))
        self.initial_belief = float(
            payload.get("initial_belief", self.initial_belief)
        )
        self.prior_belief = float(payload.get("prior_belief", self.prior_belief))
        self.stubbornness = float(payload.get("stubbornness", self.stubbornness))
        self.activity_level = float(payload.get("activity_level", self.activity_level))
        self.susceptibility = float(payload.get("susceptibility", self.susceptibility))
        self.expression_threshold = float(
            payload.get("expression_threshold", self.expression_threshold)
        )
        self.repost_tendency = float(
            payload.get("repost_tendency", self.repost_tendency)
        )
        self.policy_relevance = float(
            payload.get("policy_relevance", self.policy_relevance)
        )
        self.relevance_type = str(payload.get("relevance_type", self.relevance_type))
        self.affected_domain = str(
            payload.get("affected_domain", self.affected_domain)
        )
        self.enforcement_pressure = float(
            payload.get("enforcement_pressure", self.enforcement_pressure)
        )
        self.is_active = bool(payload.get("is_active", self.is_active))
        self.activation_tier = str(
            payload.get("activation_tier", self.activation_tier)
        )
        self.influence_tier = str(payload.get("influence_tier", self.influence_tier))
        self.cross_role_reach = str(
            payload.get("cross_role_reach", self.cross_role_reach)
        )
        self.voice_style = str(payload.get("voice_style", self.voice_style))
        self.org_context = _safe_dict(payload.get("org_context"))
        self.scenario_context = _safe_dict(payload.get("scenario_context"))
        self.init_trace = _safe_dict(payload.get("init_trace"))
        reaction = payload.get("reaction")
        if reaction:
            self.reaction = ReactionResult.from_dict(reaction)

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "role_name": self.role_name,
            "color": self.color,
            "agent_type": self.agent_type,
            "template": _safe_dict(self.template),
            "system_prompt": self.system_prompt,
            "profile": _safe_dict(self.profile),
            "concerns": _safe_list(self.concerns),
            "characteristics": _safe_dict(self.characteristics),
            "llm_agent": self.llm_agent,
            "reaction": self.reaction.to_dict() if self.reaction else None,
            "belief": float(self.belief),
            "initial_belief": float(self.initial_belief),
            "prior_belief": float(self.prior_belief),
            "stubbornness": float(self.stubbornness),
            "activity_level": float(self.activity_level),
            "susceptibility": float(self.susceptibility),
            "expression_threshold": float(self.expression_threshold),
            "repost_tendency": float(self.repost_tendency),
            "policy_relevance": float(self.policy_relevance),
            "relevance_type": self.relevance_type,
            "affected_domain": self.affected_domain,
            "enforcement_pressure": float(self.enforcement_pressure),
            "is_active": self.is_active,
            "activation_tier": self.activation_tier,
            "influence_tier": self.influence_tier,
            "cross_role_reach": self.cross_role_reach,
            "voice_style": self.voice_style,
            "org_context": _safe_dict(self.org_context),
            "scenario_context": _safe_dict(self.scenario_context),
            "init_trace": _safe_dict(self.init_trace),
            "user_id": self.user_id,
        }

    def to_public_dict(self) -> Dict[str, Any]:
        payload = self.to_runtime_dict()
        payload.pop("llm_agent", None)
        return payload


@dataclass
class PropagationStep:
    step: int
    beliefs: Dict[str, float]  # agent_id -> belief
    network_snapshot: Dict[str, Any]
    forum_stats: Dict[str, Any]
    statistics: Dict[str, Any] = field(default_factory=dict)
    influence_trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "beliefs": self.beliefs,
            "network": self.network_snapshot,
            "forum_stats": self.forum_stats,
            "statistics": self.statistics,
            "influence_trace": self.influence_trace,
        }


@dataclass
class SimulationSession:
    session_id: str
    model_type: str = "ollama"
    agents: List[PolicyAgent] = field(default_factory=list)
    policy_content: Optional[str] = None
    reaction_statistics: Dict[str, Any] = field(default_factory=dict)
    propagation_result: Dict[str, Any] = field(default_factory=dict)
    agent_summary: Dict[str, Any] = field(default_factory=dict)
    scene_summary: Dict[str, Any] = field(default_factory=dict)
    init_diagnostics: Dict[str, Any] = field(default_factory=dict)
    forum_db_path: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
