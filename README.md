# RegPolicyLab

多智能体政策影响模拟系统 - 模拟会计政策发布后不同角色的反应与网络传播

## 项目概述

本项目是一个基于 LLM 的多智能体系统，用于模拟会计政策发布后，不同利益相关者（会计师、审计师、管理层、监管者、投资者）的反应和社交网络传播过程。

## 核心功能

- **智能体初始化**: 基于角色模板和画像数据库创建个性化智能体
- **政策反应生成**: 使用 LLM 生成各角色对政策的反应
- **社交网络传播**: 模拟信念在社交网络中的传播过程
- **论坛互动系统**: 模拟政策论坛中的讨论和互动

## 项目结构

```
├── Agent.py                 # LLM 调用封装
├── app.py                   # Flask Web 应用入口
├── simulation/
│   ├── agent_pool.py        # 智能体池初始化
│   ├── engine.py            # 模拟引擎
│   ├── models.py            # 数据模型定义
│   ├── policy_config.py     # 配置加载
│   ├── services.py          # 业务服务层
│   ├── social_network.py    # 社交网络传播
│   └── role_templates/      # 角色模板 (YAML)
├── util/
│   ├── RoleProfileDB.py     # 角色画像数据库
│   └── PolicyForumDB.py     # 政策论坛数据库
├── config/                  # 配置文件
├── data/                    # 数据文件
└── tests/                   # 测试文件
```

## 角色体系

| 角色 | 描述 | 模板类型 |
|------|------|----------|
| accountant | 会计师 | conservative, detail_focused, efficiency_focused, moderate, aggressive |
| auditor | 审计师 | strict, risk_auditor, pragmatic |
| manager | 管理层 | profit_focused, strategic, risk_averse |
| regulator | 监管者 | enforcer, guide, compliance_focused |
| investor | 投资者 | value_long_term, sentiment_trader, event_driven, policy_arb |

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python app.py
```

## 依赖

- Python 3.8+
- Flask
- Anthropic SDK / OpenAI SDK
- SQLite3