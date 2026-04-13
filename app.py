"""
Flask 主入口 - 会话化交互式会计政策模拟系统

提供 API 端点用于：
1. 初始化智能体会话
2. 生成政策反应
3. 模拟社交网络传播
4. 查询论坛与画像结果
"""

from __future__ import annotations

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
    _FLASK_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    Flask = None
    jsonify = None
    request = None
    send_from_directory = None
    CORS = None
    _FLASK_IMPORT_ERROR = exc

from simulation.engine import SimulationEngine
from simulation.services import ForumService
from simulation.session_store import SimulationSessionStore


def _missing_dependency_message() -> str:
    missing = getattr(_FLASK_IMPORT_ERROR, "name", "unknown")
    return (
        f"缺少运行依赖 `{missing}`，后端服务无法启动。"
        "请先在项目根目录执行 `python3 -m pip install -r requirements.txt`。"
    )


class _DependencyPlaceholderApp:
    def route(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


if Flask is None:
    app = _DependencyPlaceholderApp()
else:
    app = Flask(__name__, static_folder=".", static_url_path="")
    CORS(app)

session_store = SimulationSessionStore()


def _resolve_session(payload=None, allow_latest=True):
    simulation_id = None
    if isinstance(payload, dict):
        simulation_id = payload.get("simulation_id")
    elif payload is not None:
        simulation_id = payload
    else:
        simulation_id = request.args.get("simulation_id")

    return session_store.get(simulation_id, allow_latest=allow_latest)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory("assets", filename)


@app.route("/api/simulate/init", methods=["POST"])
def init_simulation():
    data = request.json or {}
    model_type = data.get("model", "ollama")
    agent_counts = data.get("agent_counts", {})
    roster_mode = data.get("roster_mode")
    scene = data.get("scene")
    agent_specs = data.get("agent_specs")
    policy_content = data.get("policy_content", "")

    try:
        session = session_store.create(model_type=model_type)
        engine = SimulationEngine(session)
        agents = engine.initialize_agents(
            agent_counts=agent_counts,
            roster_mode=roster_mode,
            scene=scene,
            agent_specs=agent_specs,
            policy_content=policy_content,
        )
        session_store.save(session)

        return jsonify(
            {
                "success": True,
                "simulation_id": session.session_id,
                "total_agents": len(agents),
                "agent_summary": engine.get_agent_summary(),
                "scene_summary": engine.get_scene_summary(),
                "init_diagnostics": engine.get_init_diagnostics(),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/simulate/react", methods=["POST"])
def generate_reactions():
    data = request.json or {}
    policy_content = data.get("policy_content", "")
    session = _resolve_session(data, allow_latest=True)

    if not session:
        return jsonify({"error": "请先初始化模拟"}), 400

    if not policy_content:
        return jsonify({"error": "请输入政策内容"}), 400

    try:
        engine = SimulationEngine(session)
        reactions = engine.generate_reactions(policy_content)
        session_store.save(session)

        return jsonify(
            {
                "simulation_id": session.session_id,
                "reactions": reactions,
                "statistics": engine.get_reaction_statistics(),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/simulate/spread", methods=["POST"])
def simulate_spread():
    data = request.json or {}
    session = _resolve_session(data, allow_latest=True)

    if not session:
        return jsonify({"error": "请先初始化模拟"}), 400

    try:
        engine = SimulationEngine(session)
        spread_result = engine.simulate_network_spread()
        session_store.save(session)

        return jsonify(
            {
                "simulation_id": session.session_id,
                "network_data": spread_result["network"],
                "propagation_stats": spread_result["statistics"],
                "polarization_index": spread_result["polarization"],
                "history": spread_result.get("history", []),
                "forum_posts": spread_result.get("forum_posts", []),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/agents/profiles", methods=["GET"])
def get_agent_profiles():
    session = _resolve_session(allow_latest=True)
    if not session:
        return jsonify({"error": "请先初始化模拟"}), 400

    engine = SimulationEngine(session)
    return jsonify(
        {
            "simulation_id": session.session_id,
            "agents": engine.get_agent_profiles(),
        }
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    latest = session_store.latest()
    return jsonify(
        {
            "status": "healthy",
            "latest_simulation_id": latest.session_id if latest else None,
        }
    )


@app.route("/api/forum/posts", methods=["GET"])
def get_forum_posts():
    step = request.args.get("step", default=0, type=int)
    limit = request.args.get("limit", default=20, type=int)
    session = _resolve_session(allow_latest=True)

    if not session:
        return jsonify({"posts": [], "message": "论坛尚未初始化"})

    forum_service = ForumService(session.session_id, db_path=session.forum_db_path)
    posts = forum_service.get_top_posts(step, limit)
    return jsonify(
        {
            "simulation_id": session.session_id,
            "posts": posts,
            "total": len(posts),
        }
    )


@app.route("/api/forum/all", methods=["GET"])
def get_all_forum_posts():
    session = _resolve_session(allow_latest=True)

    if not session:
        return jsonify({"posts": [], "message": "论坛尚未初始化"})

    forum_service = ForumService(session.session_id, db_path=session.forum_db_path)
    posts = forum_service.get_all_posts()
    return jsonify(
        {
            "simulation_id": session.session_id,
            "posts": posts,
            "total": len(posts),
        }
    )


@app.route("/api/forum/stats", methods=["GET"])
def get_forum_stats():
    step = request.args.get("step", default=0, type=int)
    session = _resolve_session(allow_latest=True)

    if not session:
        return jsonify({"stats": {}, "message": "论坛尚未初始化"})

    forum_service = ForumService(session.session_id, db_path=session.forum_db_path)
    stats = forum_service.get_forum_statistics(step)
    return jsonify({"simulation_id": session.session_id, "stats": stats})


if __name__ == "__main__":
    if _FLASK_IMPORT_ERROR is not None:
        raise SystemExit(_missing_dependency_message())

    print("=" * 50)
    print("交互式会计政策模拟系统")
    print("=" * 50)
    print("访问地址: http://localhost:5000")
    print("API端点:")
    print("  - POST /api/simulate/init   初始化智能体")
    print("  - POST /api/simulate/react  生成政策反应")
    print("  - POST /api/simulate/spread 模拟网络传播")
    print("  - GET  /api/agents/profiles 获取智能体资料")
    print("=" * 50)
    app.run(debug=True, port=5000)
