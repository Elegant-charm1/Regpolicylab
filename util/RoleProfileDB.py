"""
角色画像数据库管理模块

为角色智能体（会计师、审计师、管理层、监管者）提供丰富的画像数据
支持从数据库加载个性化画像，增强智能体的多样性和真实感
"""

import sqlite3
import os
import json
from typing import Any, Dict

ROLE_PROFILE_DB_PATH = "data/role_profiles.db"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def normalize_role_profile(profile: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = dict(profile or {})
    if not normalized:
        return normalized

    normalized["past_cases"] = _safe_json_list(normalized.get("past_cases"))
    normalized["experience_years"] = int(normalized.get("experience_years") or 0)
    normalized["network_influence"] = _safe_float(
        normalized.get("network_influence"), 0.5
    )
    normalized["reputation_score"] = _safe_float(
        normalized.get("reputation_score"), 0.5
    )
    return normalized


def build_profile_org_hints(
    role: str, profile_type: str, profile: Dict[str, Any] | None
) -> Dict[str, str]:
    normalized = normalize_role_profile(profile)
    organization_type = str(normalized.get("organization_type", ""))
    organization_text = organization_type.lower()
    experience_years = int(normalized.get("experience_years", 0))

    size = "medium"
    if any(token in organization_text for token in ["四大", "证监会", "财政部", "大型", "上市", "集团"]):
        size = "large"
    elif any(token in organization_text for token in ["中型", "区域", "地方", "精品", "创业"]):
        size = "medium"
    elif any(token in organization_text for token in ["中小", "协会", "培训", "初创"]):
        size = "small"

    regulatory_exposure = {
        "accountant": "high",
        "auditor": "high",
        "manager": "medium",
        "regulator": "high",
    }.get(role, "medium")
    if any(token in organization_text for token in ["金融", "保险", "银行", "上市", "资本市场"]):
        regulatory_exposure = "high"
    elif any(token in organization_text for token in ["中小", "协会", "培训"]):
        regulatory_exposure = "medium"

    decision_authority = {
        "accountant": "medium",
        "auditor": "medium",
        "manager": "high",
        "regulator": "high",
    }.get(role, "medium")
    if any(token in organization_text for token in ["技术部", "共享中心", "培训", "咨询"]):
        decision_authority = "low"
    elif any(
        token in organization_text
        for token in ["委员会", "高管", "负责人", "证监会", "财政部", "检查局", "监管中心"]
    ):
        decision_authority = "high"

    accountability_pressure = {
        "accountant": "high",
        "auditor": "high",
        "manager": "high",
        "regulator": "high",
    }.get(role, "medium")
    if any(token in organization_text for token in ["协会", "培训", "服务"]):
        accountability_pressure = "medium"
    if experience_years >= 15 and accountability_pressure == "medium":
        accountability_pressure = "high"

    resource_constraint = "medium"
    if size == "large":
        resource_constraint = "low"
    elif size == "small":
        resource_constraint = "high"
    if profile_type in {"efficiency_focused", "pragmatic", "guide"} and size != "large":
        resource_constraint = "medium"

    return {
        "organization_type": organization_type or "相关机构",
        "organization_size": size,
        "regulatory_exposure": regulatory_exposure,
        "decision_authority": decision_authority,
        "accountability_pressure": accountability_pressure,
        "resource_constraint": resource_constraint,
    }


def init_role_profiles_db(db_path=ROLE_PROFILE_DB_PATH):
    """
    初始化角色画像数据库

    创建数据表并预填充多样化画像数据
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 如果数据库已存在，先删除
    if os.path.exists(db_path):
        os.remove(db_path)

    with sqlite3.connect(db_path) as conn:
        # 创建角色画像表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS role_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,           -- accountant/auditor/manager/regulator
                type TEXT NOT NULL,           -- conservative/moderate/aggressive等
                name TEXT NOT NULL,
                experience_years INTEGER,     -- 工作年限
                organization_type TEXT,       -- 服务机构类型
                specialization TEXT,          -- 专业领域
                risk_attitude TEXT,           -- 风险态度
                decision_style TEXT,          -- 决策风格
                industry_focus TEXT,          -- 关注行业
                past_cases TEXT,              -- 经典案例经历(JSON)
                reputation_score REAL,        -- 声誉评分(0-1)
                network_influence REAL,       -- 网络影响力系数(0-1)
                education_background TEXT,    -- 教育背景
                certification TEXT            -- 专业资质
            )
        """)

        # 预填充画像数据
        _populate_accountant_profiles(conn)
        _populate_auditor_profiles(conn)
        _populate_manager_profiles(conn)
        _populate_regulator_profiles(conn)

        conn.commit()

    return db_path


def _populate_accountant_profiles(conn):
    """填充会计师画像数据"""
    profiles = [
        {
            "role": "accountant", "type": "conservative", "name": "张稳健",
            "experience_years": 20, "organization_type": "大型国企财务部",
            "specialization": "财务报表编制、成本核算", "risk_attitude": "高度规避",
            "decision_style": "严格遵循准则", "industry_focus": "制造业、能源",
            "past_cases": json.dumps(["主导某大型国企会计准则转换项目", "参与制定行业会计核算规范"]),
            "reputation_score": 0.9, "network_influence": 0.7,
            "education_background": "财经大学会计学博士", "certification": "CPA、CFA"
        },
        {
            "role": "accountant", "type": "conservative", "name": "李谨慎",
            "experience_years": 15, "organization_type": "会计师事务所",
            "specialization": "审计配合、内控设计", "risk_attitude": "中度规避",
            "decision_style": "证据导向决策", "industry_focus": "金融、保险",
            "past_cases": json.dumps(["协助多家金融机构完成准则切换", "设计财务内控体系"]),
            "reputation_score": 0.85, "network_influence": 0.6,
            "education_background": "会计学硕士", "certification": "CPA"
        },
        {
            "role": "accountant", "type": "moderate", "name": "王平衡",
            "experience_years": 12, "organization_type": "中型民企财务部",
            "specialization": "成本优化、税务筹划", "risk_attitude": "中立平衡",
            "decision_style": "合规与效率兼顾", "industry_focus": "消费品、零售",
            "past_cases": json.dumps(["优化存货核算流程提升效率30%", "设计弹性成本分摊方案"]),
            "reputation_score": 0.75, "network_influence": 0.5,
            "education_background": "财务管理硕士", "certification": "CPA、税务师"
        },
        {
            "role": "accountant", "type": "moderate", "name": "陈务实",
            "experience_years": 8, "organization_type": "科技企业财务部",
            "specialization": "研发费用核算、收入确认", "risk_attitude": "灵活适应",
            "decision_style": "实质重于形式", "industry_focus": "科技、互联网",
            "past_cases": json.dumps(["设计研发费用归集方案", "优化收入确认时点"]),
            "reputation_score": 0.7, "network_influence": 0.55,
            "education_background": "会计学本科", "certification": "CPA"
        },
        {
            "role": "accountant", "type": "detail_focused", "name": "许条文",
            "experience_years": 14, "organization_type": "大型会计师事务所技术部",
            "specialization": "准则解读、会计政策研究", "risk_attitude": "中度规避",
            "decision_style": "逐条核对判断", "industry_focus": "上市公司、制造业",
            "past_cases": json.dumps(["参与复杂准则应用意见起草", "为大型集团提供条款适用分析"]),
            "reputation_score": 0.86, "network_influence": 0.58,
            "education_background": "会计学博士", "certification": "CPA"
        },
        {
            "role": "accountant", "type": "efficiency_focused", "name": "何提效",
            "experience_years": 9, "organization_type": "平台型互联网企业财务共享中心",
            "specialization": "流程自动化、核算优化", "risk_attitude": "务实平衡",
            "decision_style": "效率优先落地", "industry_focus": "互联网、零售",
            "past_cases": json.dumps(["主导月结流程自动化改造", "压缩政策切换后的财务关账周期"]),
            "reputation_score": 0.72, "network_influence": 0.62,
            "education_background": "财务管理硕士", "certification": "CPA、信息系统项目管理师"
        },
        {
            "role": "accountant", "type": "aggressive", "name": "赵创新",
            "experience_years": 10, "organization_type": "投资公司财务部",
            "specialization": "金融工具核算、公允价值", "risk_attitude": "适度冒险",
            "decision_style": "准则灵活性利用", "industry_focus": "投资、金融",
            "past_cases": json.dumps(["设计复杂金融工具核算方案", "优化投资组合价值计量"]),
            "reputation_score": 0.8, "network_influence": 0.65,
            "education_background": "金融学硕士", "certification": "CPA、ACCA"
        },
        {
            "role": "accountant", "type": "aggressive", "name": "钱进取",
            "experience_years": 6, "organization_type": "创业公司财务部",
            "specialization": "股权激励核算、并购会计", "risk_attitude": "创新导向",
            "decision_style": "价值最大化", "industry_focus": "初创企业、科技",
            "past_cases": json.dumps(["设计多轮股权激励核算方案", "参与并购会计处理"]),
            "reputation_score": 0.65, "network_influence": 0.45,
            "education_background": "MBA", "certification": "CPA"
        }
    ]

    for profile in profiles:
        conn.execute("""
            INSERT INTO role_profiles (
                role, type, name, experience_years, organization_type,
                specialization, risk_attitude, decision_style, industry_focus,
                past_cases, reputation_score, network_influence,
                education_background, certification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile["role"], profile["type"], profile["name"],
            profile["experience_years"], profile["organization_type"],
            profile["specialization"], profile["risk_attitude"],
            profile["decision_style"], profile["industry_focus"],
            profile["past_cases"], profile["reputation_score"],
            profile["network_influence"], profile["education_background"],
            profile["certification"]
        ))


def _populate_auditor_profiles(conn):
    """填充审计师画像数据"""
    profiles = [
        {
            "role": "auditor", "type": "strict", "name": "孙严格",
            "experience_years": 18, "organization_type": "四大会计师事务所",
            "specialization": "风险评估、舞弊检查", "risk_attitude": "高度警惕",
            "decision_style": "证据充分性优先", "industry_focus": "上市公司、金融",
            "past_cases": json.dumps(["主导重大舞弊案审计", "设计行业专项审计程序"]),
            "reputation_score": 0.95, "network_influence": 0.85,
            "education_background": "会计学博士", "certification": "CPA、CIA"
        },
        {
            "role": "auditor", "type": "strict", "name": "周审慎",
            "experience_years": 12, "organization_type": "大型会计师事务所",
            "specialization": "内控审计、合规检查", "risk_attitude": "中度警惕",
            "decision_style": "风险导向审计", "industry_focus": "制造业、房地产",
            "past_cases": json.dumps(["发现多起内控缺陷", "优化审计风险模型"]),
            "reputation_score": 0.88, "network_influence": 0.7,
            "education_background": "审计学硕士", "certification": "CPA"
        },
        {
            "role": "auditor", "type": "pragmatic", "name": "吴务实",
            "experience_years": 10, "organization_type": "中型会计师事务所",
            "specialization": "效率审计、实质审查", "risk_attitude": "平衡风险",
            "decision_style": "重大风险优先", "industry_focus": "中小企业、服务业",
            "past_cases": json.dumps(["设计高效审计流程", "优化抽样方法"]),
            "reputation_score": 0.8, "network_influence": 0.6,
            "education_background": "会计学硕士", "certification": "CPA"
        },
        {
            "role": "auditor", "type": "pragmatic", "name": "郑协调",
            "experience_years": 8, "organization_type": "区域会计师事务所",
            "specialization": "沟通协调、客户服务", "risk_attitude": "务实灵活",
            "decision_style": "风险与效率兼顾", "industry_focus": "地方企业、政府",
            "past_cases": json.dumps(["成功协调多项审计争议", "建立客户沟通机制"]),
            "reputation_score": 0.75, "network_influence": 0.5,
            "education_background": "会计学本科", "certification": "CPA"
        },
        {
            "role": "auditor", "type": "risk_auditor", "name": "顾预警",
            "experience_years": 11, "organization_type": "大型会计师事务所风险咨询部",
            "specialization": "风险建模、异常识别", "risk_attitude": "高度警惕",
            "decision_style": "模型驱动预警", "industry_focus": "金融、科技",
            "past_cases": json.dumps(["搭建收入确认风险预警模型", "主导异常交易识别专项审计"]),
            "reputation_score": 0.84, "network_influence": 0.66,
            "education_background": "审计学硕士", "certification": "CPA、FRM"
        },
        {
            "role": "auditor", "type": "risk_auditor", "name": "林监测",
            "experience_years": 7, "organization_type": "行业精品审计机构",
            "specialization": "内控监测、风险评估", "risk_attitude": "中度警惕",
            "decision_style": "持续监控判断", "industry_focus": "制造业、消费品",
            "past_cases": json.dumps(["建立集团内控风险雷达", "优化高风险样本抽取规则"]),
            "reputation_score": 0.76, "network_influence": 0.54,
            "education_background": "会计学本科", "certification": "CPA、CIA"
        }
    ]

    for profile in profiles:
        conn.execute("""
            INSERT INTO role_profiles (
                role, type, name, experience_years, organization_type,
                specialization, risk_attitude, decision_style, industry_focus,
                past_cases, reputation_score, network_influence,
                education_background, certification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile["role"], profile["type"], profile["name"],
            profile["experience_years"], profile["organization_type"],
            profile["specialization"], profile["risk_attitude"],
            profile["decision_style"], profile["industry_focus"],
            profile["past_cases"], profile["reputation_score"],
            profile["network_influence"], profile["education_background"],
            profile["certification"]
        ))


def _populate_manager_profiles(conn):
    """填充管理层画像数据"""
    profiles = [
        {
            "role": "manager", "type": "profit_focused", "name": "冯业绩",
            "experience_years": 15, "organization_type": "上市公司高管",
            "specialization": "市值管理、业绩优化", "risk_attitude": "业绩导向",
            "decision_style": "市场预期优先", "industry_focus": "消费品、科技",
            "past_cases": json.dumps(["主导多次业绩提升项目", "优化信息披露策略"]),
            "reputation_score": 0.85, "network_influence": 0.75,
            "education_background": "MBA、金融硕士", "certification": "无"
        },
        {
            "role": "manager", "type": "profit_focused", "name": "褚市值",
            "experience_years": 10, "organization_type": "民企财务总监",
            "specialization": "融资规划、投资者关系", "risk_attitude": "市值优先",
            "decision_style": "投资者预期管理", "industry_focus": "成长型企业",
            "past_cases": json.dumps(["成功完成多轮融资", "建立投资者沟通体系"]),
            "reputation_score": 0.78, "network_influence": 0.65,
            "education_background": "财务管理硕士", "certification": "CPA"
        },
        {
            "role": "manager", "type": "risk_averse", "name": "卫稳健",
            "experience_years": 20, "organization_type": "国企财务负责人",
            "specialization": "合规管理、风险控制", "risk_attitude": "高度规避",
            "decision_style": "合规风险优先", "industry_focus": "国企、公用事业",
            "past_cases": json.dumps(["建立完善内控体系", "主导合规整改项目"]),
            "reputation_score": 0.9, "network_influence": 0.7,
            "education_background": "会计学硕士", "certification": "CPA、高级会计师"
        },
        {
            "role": "manager", "type": "risk_averse", "name": "蒋审慎",
            "experience_years": 12, "organization_type": "金融机构财务总监",
            "specialization": "监管合规、风险管理", "risk_attitude": "中度规避",
            "decision_style": "长期稳定优先", "industry_focus": "金融、银行",
            "past_cases": json.dumps(["应对监管检查零违规", "建立风险预警机制"]),
            "reputation_score": 0.82, "network_influence": 0.6,
            "education_background": "金融学硕士", "certification": "CPA、CFA"
        },
        {
            "role": "manager", "type": "strategic", "name": "曹远见",
            "experience_years": 18, "organization_type": "上市公司战略与财务委员会",
            "specialization": "战略规划、资本配置", "risk_attitude": "长期均衡",
            "decision_style": "全局权衡决策", "industry_focus": "高端制造、科技",
            "past_cases": json.dumps(["主导准则变化下的业务重组评估", "推动财务政策与战略转型协同"]),
            "reputation_score": 0.87, "network_influence": 0.78,
            "education_background": "MBA、战略管理硕士", "certification": "CFA"
        },
        {
            "role": "manager", "type": "strategic", "name": "韩布局",
            "experience_years": 13, "organization_type": "产业集团投资管理部",
            "specialization": "产业布局、投融资协同", "risk_attitude": "机会识别导向",
            "decision_style": "战略机会优先", "industry_focus": "新能源、产业投资",
            "past_cases": json.dumps(["在监管变化中调整资本开支节奏", "设计长期对外沟通与资源配置方案"]),
            "reputation_score": 0.8, "network_influence": 0.69,
            "education_background": "金融学硕士", "certification": "CPA、CFA"
        }
    ]

    for profile in profiles:
        conn.execute("""
            INSERT INTO role_profiles (
                role, type, name, experience_years, organization_type,
                specialization, risk_attitude, decision_style, industry_focus,
                past_cases, reputation_score, network_influence,
                education_background, certification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile["role"], profile["type"], profile["name"],
            profile["experience_years"], profile["organization_type"],
            profile["specialization"], profile["risk_attitude"],
            profile["decision_style"], profile["industry_focus"],
            profile["past_cases"], profile["reputation_score"],
            profile["network_influence"], profile["education_background"],
            profile["certification"]
        ))


def _populate_regulator_profiles(conn):
    """填充监管者画像数据"""
    profiles = [
        {
            "role": "regulator", "type": "enforcer", "name": "沈执法",
            "experience_years": 15, "organization_type": "证监会/财政部",
            "specialization": "违规查处、政策执行", "risk_attitude": "高压态势",
            "decision_style": "严格执法", "industry_focus": "资本市场",
            "past_cases": json.dumps(["主导多起重大违规查处", "制定执法标准"]),
            "reputation_score": 0.92, "network_influence": 0.9,
            "education_background": "法学博士", "certification": "司法资格"
        },
        {
            "role": "regulator", "type": "enforcer", "name": "韩监管",
            "experience_years": 10, "organization_type": "交易所/协会",
            "specialization": "信息披露监管、自律管理", "risk_attitude": "主动监管",
            "decision_style": "预防性监管", "industry_focus": "上市公司",
            "past_cases": json.dumps(["建立信息披露预警系统", "设计自律监管机制"]),
            "reputation_score": 0.85, "network_influence": 0.75,
            "education_background": "会计学硕士", "certification": "CPA"
        },
        {
            "role": "regulator", "type": "guide", "name": "杨指导",
            "experience_years": 12, "organization_type": "财政部/准则委",
            "specialization": "准则解读、市场教育", "risk_attitude": "指导导向",
            "decision_style": "沟通教育优先", "industry_focus": "全市场",
            "past_cases": json.dumps(["主导准则培训项目", "编写解读材料"]),
            "reputation_score": 0.88, "network_influence": 0.8,
            "education_background": "会计学博士", "certification": "CPA"
        },
        {
            "role": "regulator", "type": "guide", "name": "朱服务",
            "experience_years": 8, "organization_type": "协会/培训中心",
            "specialization": "实务指导、答疑解惑", "risk_attitude": "服务导向",
            "decision_style": "反馈收集改进", "industry_focus": "中小企业",
            "past_cases": json.dumps(["建立咨询答疑平台", "组织实务培训"]),
            "reputation_score": 0.8, "network_influence": 0.65,
            "education_background": "会计学硕士", "certification": "CPA"
        },
        {
            "role": "regulator", "type": "compliance_focused", "name": "严核查",
            "experience_years": 16, "organization_type": "证监会检查局",
            "specialization": "合规检查、违规认定", "risk_attitude": "零容忍",
            "decision_style": "程序规范优先", "industry_focus": "资本市场、上市公司",
            "past_cases": json.dumps(["主导多轮信息披露专项检查", "制定合规核查重点清单"]),
            "reputation_score": 0.91, "network_influence": 0.83,
            "education_background": "法学硕士", "certification": "司法资格"
        },
        {
            "role": "regulator", "type": "compliance_focused", "name": "邵督导",
            "experience_years": 11, "organization_type": "交易所监管中心",
            "specialization": "规则执行、持续督导", "risk_attitude": "主动监管",
            "decision_style": "检查闭环管理", "industry_focus": "成长型企业、信息披露",
            "past_cases": json.dumps(["推动高风险企业完成限期整改", "完善持续督导过程记录标准"]),
            "reputation_score": 0.84, "network_influence": 0.72,
            "education_background": "会计学硕士", "certification": "CPA"
        }
    ]

    for profile in profiles:
        conn.execute("""
            INSERT INTO role_profiles (
                role, type, name, experience_years, organization_type,
                specialization, risk_attitude, decision_style, industry_focus,
                past_cases, reputation_score, network_influence,
                education_background, certification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile["role"], profile["type"], profile["name"],
            profile["experience_years"], profile["organization_type"],
            profile["specialization"], profile["risk_attitude"],
            profile["decision_style"], profile["industry_focus"],
            profile["past_cases"], profile["reputation_score"],
            profile["network_influence"], profile["education_background"],
            profile["certification"]
        ))


def load_role_profile(role, profile_type, db_path=ROLE_PROFILE_DB_PATH):
    """
    加载指定角色和类型的画像

    Args:
        role: 角色类型 (accountant/auditor/manager/regulator)
        profile_type: 画像类型 (conservative/moderate/aggressive等)
        db_path: 数据库路径

    Returns:
        dict: 画像数据，如果未找到则返回该角色的任意画像
    """
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        # 首先尝试精确匹配
        cursor = conn.execute("""
            SELECT * FROM role_profiles
            WHERE role = ? AND type = ?
            ORDER BY RANDOM()
            LIMIT 1
        """, (role, profile_type))

        row = cursor.fetchone()
        if row:
            return normalize_role_profile(dict(row))

        # 如果精确匹配失败，回退到该角色的任意画像
        cursor = conn.execute("""
            SELECT * FROM role_profiles
            WHERE role = ?
            ORDER BY RANDOM()
            LIMIT 1
        """, (role,))

        row = cursor.fetchone()
        if row:
            return normalize_role_profile(dict(row))

    return None


def get_all_profiles_for_role(role, db_path=ROLE_PROFILE_DB_PATH):
    """
    获取指定角色的所有画像

    Args:
        role: 角色类型
        db_path: 数据库路径

    Returns:
        list: 画像列表
    """
    if not os.path.exists(db_path):
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT * FROM role_profiles WHERE role = ?
        """, (role,))

        return [normalize_role_profile(dict(row)) for row in cursor.fetchall()]


# 初始化数据库（首次导入时自动执行）
if __name__ == "__main__":
    db_path = init_role_profiles_db()
    print(f"角色画像数据库已创建: {db_path}")

    # 验证数据
    for role in ["accountant", "auditor", "manager", "regulator"]:
        profiles = get_all_profiles_for_role(role)
        print(f"{role}: {len(profiles)} 个画像")
