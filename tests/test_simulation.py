"""
核心功能测试模块

测试范围：
- AgentPool 初始化
- SimulationEngine 智能体初始化
- ReactionService 反应生成 (mock LLM)
- BaseAgent API调用 (mock)
- ReactionParser JSON解析
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestReactionParser(unittest.TestCase):
    """测试反应解析器"""

    def setUp(self):
        from simulation.services import ReactionParser
        from simulation.models import ReactionResult
        self.parser = ReactionParser()
        self.fallback = ReactionResult()

    def test_parse_valid_json_in_fence(self):
        """测试解析 fenced JSON"""
        content = '''
```json
{
    "understanding": "政策涉及收入确认方式变更",
    "decision_stance": "保守",
    "impact_level": "高",
    "concerns_analysis": "关注合规风险",
    "recommendation": "等待监管明确",
    "decision_reason": "风险规避",
    "belief_score": -0.5
}
```
'''
        result = self.parser.parse(content, self.fallback)
        self.assertEqual(result.decision_stance, "保守")
        self.assertEqual(result.belief_score, -0.5)

    def test_parse_bare_json(self):
        """测试解析裸 JSON"""
        content = '{"understanding": "测试", "decision_stance": "激进", "belief_score": 0.8}'
        result = self.parser.parse(content, self.fallback)
        self.assertEqual(result.decision_stance, "激进")
        self.assertEqual(result.belief_score, 0.8)

    def test_parse_belief_score_clamp(self):
        """测试信念值边界限制"""
        content = '{"belief_score": 2.0}'  # 超出范围
        result = self.parser.parse(content, self.fallback)
        self.assertEqual(result.belief_score, 1.0)  # 应被限制到1.0

        content = '{"belief_score": -2.0}'  # 超出范围
        result = self.parser.parse(content, self.fallback)
        self.assertEqual(result.belief_score, -1.0)

    def test_parse_invalid_json_returns_fallback(self):
        """测试无效JSON返回fallback"""
        content = "这不是JSON"
        result = self.parser.parse(content, self.fallback)
        self.assertEqual(result.decision_stance, self.fallback.decision_stance)


class TestReactionResult(unittest.TestCase):
    """测试 ReactionResult 数据模型"""

    def test_from_dict(self):
        """测试从字典创建"""
        from simulation.models import ReactionResult
        data = {
            "understanding": "政策分析",
            "decision_stance": "中立",
            "impact_level": "中",
            "belief_score": 0.0
        }
        result = ReactionResult.from_dict(data)
        self.assertEqual(result.understanding, "政策分析")

    def test_to_dict(self):
        """测试转换为字典"""
        from simulation.models import ReactionResult
        result = ReactionResult(
            understanding="测试理解",
            decision_stance="保守",
            impact_level="高",
            concerns_analysis="风险分析",
            recommendation="建议等待",
            decision_reason="谨慎处理",
            belief_score=-0.3
        )
        data = result.to_dict()
        self.assertEqual(data["belief_score"], -0.3)
        self.assertIn("understanding", data)


class TestPolicyAgent(unittest.TestCase):
    """测试 PolicyAgent 数据模型"""

    def test_apply_reaction(self):
        """测试应用反应"""
        from simulation.models import PolicyAgent, ReactionResult
        agent = PolicyAgent(
            id="test_001",
            role="accountant",
            role_name="会计师",
            color="#3b82f6"
        )
        reaction = ReactionResult(belief_score=0.6)
        agent.apply_reaction(reaction)
        self.assertEqual(agent.belief, 0.6)
        self.assertEqual(agent.initial_belief, 0.6)

    def test_to_public_dict_excludes_llm_agent(self):
        """测试公开字典不包含llm_agent"""
        from simulation.models import PolicyAgent
        mock_llm = MagicMock()
        agent = PolicyAgent(
            id="test_001",
            role="auditor",
            role_name="审计师",
            color="#10b981",
            llm_agent=mock_llm
        )
        public_dict = agent.to_public_dict()
        self.assertNotIn("llm_agent", public_dict)


class TestReactionPromptBuilder(unittest.TestCase):
    """测试反应提示词构建器"""

    def test_build_includes_concerns(self):
        """测试提示词包含关注点"""
        from simulation.services import ReactionPromptBuilder
        from simulation.models import PolicyAgent
        builder = ReactionPromptBuilder()
        agent = PolicyAgent(
            id="test_001",
            role="accountant",
            role_name="会计师",
            color="#3b82f6",
            concerns=["合规风险", "审计风险"],
            system_prompt="你是一位会计师"
        )
        prompt = builder.build(agent, "新会计准则发布")
        self.assertIn("合规风险", prompt)
        self.assertIn("审计风险", prompt)
        self.assertIn("会计师", prompt)


class TestReactionService(unittest.TestCase):
    """测试反应生成服务"""

    def test_generate_single_reaction_with_mock_llm(self):
        """测试单个反应生成 (mock LLM)"""
        from simulation.services import ReactionService
        from simulation.models import PolicyAgent
        service = ReactionService()

        # 创建 mock LLM agent
        mock_llm = MagicMock()
        mock_llm.get_response.return_value = {
            "response": json.dumps({
                "understanding": "政策涉及会计处理变更",
                "decision_stance": "保守",
                "impact_level": "高",
                "concerns_analysis": "关注合规风险",
                "recommendation": "等待监管明确",
                "decision_reason": "风险规避",
                "belief_score": -0.5
            })
        }

        agent = PolicyAgent(
            id="test_001",
            role="accountant",
            role_name="会计师",
            color="#3b82f6",
            llm_agent=mock_llm
        )

        reaction = service.generate_single_reaction(agent, "新政策内容")
        self.assertEqual(reaction.decision_stance, "保守")
        self.assertEqual(reaction.belief_score, -0.5)

    def test_generate_reactions_batch(self):
        """测试批量反应生成"""
        from simulation.services import ReactionService
        from simulation.models import PolicyAgent
        service = ReactionService()

        mock_llm = MagicMock()
        mock_llm.get_response.return_value = {
            "response": json.dumps({
                "decision_stance": "中立",
                "belief_score": 0.0
            })
        }

        agents = [
            PolicyAgent(id="a1", role="accountant", role_name="会计师", color="#3b82f6", llm_agent=mock_llm),
            PolicyAgent(id="a2", role="auditor", role_name="审计师", color="#10b981", llm_agent=mock_llm),
        ]

        reactions, stats = service.generate_reactions(agents, "政策内容")
        self.assertEqual(len(reactions), 2)
        self.assertIn("stance_distribution", stats)


class TestForumService(unittest.TestCase):
    """测试论坛服务"""

    def test_build_db_path(self):
        """测试数据库路径构建"""
        from simulation.services import ForumService
        path = ForumService.build_db_path("test-session-123")
        self.assertIn("policy_forum_test-session-123.db", path)


class TestPolicySimulationConfig(unittest.TestCase):
    """测试政策模拟配置"""

    def test_get_config_returns_dict(self):
        """测试配置获取返回字典"""
        from simulation.policy_config import get_policy_simulation_config
        config = get_policy_simulation_config()
        self.assertIsInstance(config, dict)
        self.assertIn("agent_traits", config)

    def test_role_social_traits_present(self):
        """测试角色社交特征存在"""
        from simulation.policy_config import get_policy_simulation_config
        config = get_policy_simulation_config()
        traits = config.get("agent_traits", {}).get("role_social_traits", {})
        self.assertIn("accountant", traits)
        self.assertIn("stubbornness", traits["accountant"])


class TestAgentPool(unittest.TestCase):
    """测试智能体池"""

    def test_role_names_defined(self):
        """测试角色名称定义"""
        from simulation.agent_pool import AgentPool
        self.assertIn("accountant", AgentPool.ROLE_NAMES)
        self.assertEqual(AgentPool.ROLE_NAMES["accountant"], "会计师")

    def test_role_colors_defined(self):
        """测试角色颜色定义"""
        from simulation.agent_pool import AgentPool
        self.assertIn("accountant", AgentPool.ROLE_COLORS)

    def test_role_social_traits_from_config(self):
        """测试社交特征从配置加载"""
        from simulation.agent_pool import AgentPool
        pool = AgentPool()
        # 验证从 policy_config 加载的社交特征
        self.assertIn("accountant", pool.role_social_traits)
        traits = pool.role_social_traits["accountant"]
        self.assertIn("stubbornness", traits)


class TestSimulationSession(unittest.TestCase):
    """测试模拟会话"""

    def test_session_creation(self):
        """测试会话创建"""
        from simulation.models import SimulationSession
        session = SimulationSession(session_id="test-123")
        self.assertEqual(session.session_id, "test-123")
        self.assertEqual(session.model_type, "ollama")
        self.assertEqual(len(session.agents), 0)

    def test_touch_updates_timestamp(self):
        """测试touch更新时间戳"""
        from simulation.models import SimulationSession
        import time
        session = SimulationSession(session_id="test-123")
        old_time = session.updated_at
        time.sleep(0.1)
        session.touch()
        self.assertNotEqual(session.updated_at, old_time)


if __name__ == "__main__":
    unittest.main(verbosity=2)