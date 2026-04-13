"""
模拟引擎 - 会话化政策影响编排器

负责：
1. 初始化 agent 会话状态
2. 调用反应生成服务
3. 调用社交传播服务
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agent_pool import AgentPool
from .models import PolicyAgent, SimulationSession
from .services import ForumService, PropagationService, ReactionService


class SimulationEngine:
    """围绕单个 SimulationSession 的轻量编排器。"""

    def __init__(self, session: SimulationSession):
        self.session = session
        self.model_type = session.model_type
        self.reaction_service = ReactionService()
        self.forum_service = ForumService(
            session.session_id, db_path=session.forum_db_path
        )
        self.session.forum_db_path = self.forum_service.db_path
        self.propagation_service = PropagationService(self.forum_service)

    @property
    def agents(self) -> List[PolicyAgent]:
        return self.session.agents

    def initialize_agents(
        self,
        agent_counts: Optional[Dict[str, int]] = None,
        roster_mode: Optional[str] = None,
        scene: Optional[Dict[str, Any]] = None,
        agent_specs: Optional[List[Dict[str, Any]]] = None,
        policy_content: str = "",
    ) -> List[PolicyAgent]:
        agent_pool = AgentPool(self.model_type)
        agent_pool.initialize(
            agent_counts=agent_counts or {},
            roster_mode=roster_mode,
            scene=scene,
            agent_specs=agent_specs,
            policy_content=policy_content,
        )

        self.session.agents = agent_pool.agents
        self.session.agent_summary = agent_pool.get_summary()
        self.session.scene_summary = agent_pool.get_scene_summary()
        self.session.init_diagnostics = agent_pool.get_init_diagnostics()
        self.session.policy_content = None
        self.session.reaction_statistics = {}
        self.session.propagation_result = {}
        self.forum_service.reset()
        self.session.touch()

        return self.session.agents

    def generate_reactions(self, policy_content: str):
        if not self.session.agents:
            raise ValueError("智能体尚未初始化")

        self.session.policy_content = policy_content
        reactions, statistics = self.reaction_service.generate_reactions(
            self.session.agents, policy_content
        )
        self.session.reaction_statistics = statistics
        self.session.touch()
        return reactions

    def simulate_network_spread(self, steps: int = 10):
        if not self.session.agents:
            raise ValueError("智能体尚未初始化")

        propagation_result = self.propagation_service.simulate(
            self.session.agents, steps=steps
        )
        self.session.propagation_result = propagation_result
        self.session.touch()
        return propagation_result

    def get_reaction_statistics(self):
        return self.session.reaction_statistics

    def get_agent_profiles(self):
        return [agent.to_public_dict() for agent in self.session.agents]

    def get_agent_summary(self):
        return self.session.agent_summary

    def get_scene_summary(self):
        return self.session.scene_summary

    def get_init_diagnostics(self):
        return self.session.init_diagnostics
