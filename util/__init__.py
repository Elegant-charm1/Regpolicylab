"""
RegPolicyLab 工具模块

包含数据库管理、论坛互动、角色画像等工具类
"""

from .RoleProfileDB import load_role_profile, init_role_profiles_db, ROLE_PROFILE_DB_PATH
from .PolicyForumDB import (
    init_policy_forum, create_post, react_to_post,
    get_top_posts, get_forum_statistics, get_all_posts, FORUM_DB_PATH
)