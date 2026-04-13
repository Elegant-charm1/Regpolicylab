"""
政策模拟配置加载模块

从 config/policy_simulation.yaml 加载配置，提供最小 fallback。
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


# 最小 fallback 配置（仅在 YAML 文件不存在时使用）
MINIMAL_FALLBACK_CONFIG: Dict[str, Any] = {
    "agent_traits": {
        "role_social_traits": {
            "accountant": {"stubbornness": 0.55, "activity_level": 0.55, "susceptibility": 0.45},
            "auditor": {"stubbornness": 0.65, "activity_level": 0.50, "susceptibility": 0.35},
            "manager": {"stubbornness": 0.45, "activity_level": 0.60, "susceptibility": 0.50},
            "regulator": {"stubbornness": 0.75, "activity_level": 0.45, "susceptibility": 0.25},
            "investor": {"stubbornness": 0.35, "activity_level": 0.70, "susceptibility": 0.65},
        }
    },
    "initialization": {
        "default_roster_mode": "default",
        "rosters": {"default": []},
    },
    "social_network": {"default_top_k": 5},
}


@lru_cache(maxsize=1)
def get_policy_simulation_config() -> Dict[str, Any]:
    """
    加载政策模拟配置。

    优先从 config/policy_simulation.yaml 加载。
    如果文件不存在或加载失败，返回最小 fallback 配置。
    """
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "policy_simulation.yaml"

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if isinstance(loaded, dict) and loaded:
                logger.info(f"成功加载配置文件: {config_path}")
                return loaded
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}, 使用 fallback 配置")

    logger.warning(f"配置文件不存在: {config_path}, 使用 fallback 配置")
    return MINIMAL_FALLBACK_CONFIG.copy()