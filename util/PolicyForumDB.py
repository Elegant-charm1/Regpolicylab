"""
政策讨论论坛数据库 - 简化版

基于TwinMarket论坛模块简化设计，适配政策模拟场景
支持智能体发布政策观点、点赞/反对互动、热门帖子排序
"""

import sqlite3
import os
from datetime import datetime

FORUM_DB_PATH = "data/policy_forum.db"


def init_policy_forum(db_path=FORUM_DB_PATH):
    """
    初始化政策论坛数据库

    创建帖子表和互动表，如果已存在则清空数据（不删除文件以避免 Windows 文件锁冲突）
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        # 帖子表 - 智能体对政策的观点发布
        conn.execute("""
            CREATE TABLE IF NOT EXISTS policy_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                agent_type TEXT,
                content TEXT NOT NULL,
                stance TEXT,          -- 保守/中立/激进
                score INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                unlikes INTEGER DEFAULT 0,
                step INTEGER,         -- 模拟步骤
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 互动表 - 点赞/反对
        conn.execute("""
            CREATE TABLE IF NOT EXISTS policy_reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                post_id INTEGER NOT NULL,
                type TEXT CHECK(type IN ('like', 'unlike')) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES policy_posts(id)
            )
        """)

        # 清空现有数据（每次模拟重新开始），避免删除文件导致的 Windows 文件锁冲突
        conn.execute("DELETE FROM policy_reactions")
        conn.execute("DELETE FROM policy_posts")

        conn.commit()

    return db_path


def create_post(agent_id, agent_role, agent_type, content, stance, step, db_path=FORUM_DB_PATH):
    """
    智能体发布政策观点帖子

    Args:
        agent_id: 智能体ID
        agent_role: 智能体角色
        agent_type: 智能体类型
        content: 帖子内容（政策理解）
        stance: 立场（保守/中立/激进）
        step: 当前模拟步骤
        db_path: 数据库路径

    Returns:
        int: 帖子ID
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO policy_posts (agent_id, agent_role, agent_type, content, stance, step)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (agent_id, agent_role, agent_type, content, stance, step))
        conn.commit()
        return cursor.lastrowid


def react_to_post(agent_id, post_id, reaction_type, db_path=FORUM_DB_PATH):
    """
    智能体对帖子点赞/反对

    Args:
        agent_id: 智能体ID
        post_id: 帖子ID
        reaction_type: 互动类型 ('like' 或 'unlike')
        db_path: 数据库路径

    Returns:
        bool: 是否成功（已互动过的返回False）
    """
    with sqlite3.connect(db_path) as conn:
        # 检查是否已互动过
        cursor = conn.execute("""
            SELECT id FROM policy_reactions
            WHERE agent_id = ? AND post_id = ?
        """, (agent_id, post_id))

        if cursor.fetchone():
            return False  # 已互动过

        # 添加互动记录
        conn.execute("""
            INSERT INTO policy_reactions (agent_id, post_id, type)
            VALUES (?, ?, ?)
        """, (agent_id, post_id, reaction_type))

        # 更新帖子评分和计数
        delta = 1 if reaction_type == 'like' else -1
        if reaction_type == 'like':
            conn.execute("""
                UPDATE policy_posts SET score = score + ?, likes = likes + 1 WHERE id = ?
            """, (delta, post_id))
        else:
            conn.execute("""
                UPDATE policy_posts SET score = score + ?, unlikes = unlikes + 1 WHERE id = ?
            """, (delta, post_id))

        conn.commit()
        return True


def get_top_posts(step, limit=10, db_path=FORUM_DB_PATH):
    """
    获取指定步骤的热门帖子

    Args:
        step: 模拟步骤
        limit: 返回数量上限
        db_path: 数据库路径

    Returns:
        list: 帖子列表
    """
    if not os.path.exists(db_path):
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT id, agent_id, agent_role, agent_type, content, stance, score, likes, unlikes
            FROM policy_posts
            WHERE step = ?
            ORDER BY score DESC
            LIMIT ?
        """, (step, limit))

        return [dict(row) for row in cursor.fetchall()]


def get_posts_by_agent(agent_id, db_path=FORUM_DB_PATH):
    """
    获取指定智能体的所有帖子

    Args:
        agent_id: 智能体ID
        db_path: 数据库路径

    Returns:
        list: 帖子列表
    """
    if not os.path.exists(db_path):
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT * FROM policy_posts WHERE agent_id = ? ORDER BY step
        """, (agent_id,))

        return [dict(row) for row in cursor.fetchall()]


def get_agent_reactions(agent_id, db_path=FORUM_DB_PATH):
    """
    获取指定智能体的所有互动记录

    Args:
        agent_id: 智能体ID
        db_path: 数据库路径

    Returns:
        list: 互动记录列表
    """
    if not os.path.exists(db_path):
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT pr.*, pp.agent_id as post_author, pp.stance as post_stance
            FROM policy_reactions pr
            JOIN policy_posts pp ON pr.post_id = pp.id
            WHERE pr.agent_id = ?
        """, (agent_id,))

        return [dict(row) for row in cursor.fetchall()]


def get_forum_statistics(step, db_path=FORUM_DB_PATH):
    """
    获取论坛统计信息

    Args:
        step: 模拟步骤
        db_path: 数据库路径

    Returns:
        dict: 统计信息
    """
    if not os.path.exists(db_path):
        return {}

    with sqlite3.connect(db_path) as conn:
        # 帖子总数
        cursor = conn.execute("""
            SELECT COUNT(*) as total_posts,
                   COALESCE(SUM(likes), 0) as total_likes,
                   COALESCE(SUM(unlikes), 0) as total_unlikes,
                   COALESCE(AVG(score), 0) as avg_score
            FROM policy_posts WHERE step = ?
        """, (step,))
        row = cursor.fetchone()
        post_stats = {
            "total_posts": row[0] if row else 0,
            "total_likes": row[1] if row else 0,
            "total_unlikes": row[2] if row else 0,
            "avg_score": round(row[3], 2) if row and row[3] else 0
        }

        # 各立场分布
        cursor = conn.execute("""
            SELECT stance, COUNT(*) as count
            FROM policy_posts WHERE step = ?
            GROUP BY stance
        """, (step,))
        stance_dist = {row[0]: row[1] for row in cursor.fetchall()}

        # 各角色活跃度
        cursor = conn.execute("""
            SELECT agent_role, COUNT(*) as post_count
            FROM policy_posts WHERE step = ?
            GROUP BY agent_role
        """, (step,))
        role_activity = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "post_stats": post_stats,
            "stance_distribution": stance_dist,
            "role_activity": role_activity
        }


def get_all_posts(db_path=FORUM_DB_PATH):
    """
    获取所有帖子（用于前端展示）

    Args:
        db_path: 数据库路径

    Returns:
        list: 所有帖子列表
    """
    if not os.path.exists(db_path):
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT id, agent_id, agent_role, agent_type, content, stance, score, likes, unlikes, step
            FROM policy_posts
            ORDER BY step, score DESC
        """)

        return [dict(row) for row in cursor.fetchall()]


# 初始化数据库（首次导入时自动执行）
if __name__ == "__main__":
    db_path = init_policy_forum()
    print(f"政策论坛数据库已创建: {db_path}")

    # 测试功能
    post_id = create_post("accountant_001", "accountant", "conservative",
                         "新准则对存货核算影响较大，需谨慎处理", "保守", 0)
    print(f"创建帖子: {post_id}")

    react_to_post("auditor_001", post_id, "like")
    print("点赞成功")

    top_posts = get_top_posts(0)
    print(f"热门帖子: {top_posts}")