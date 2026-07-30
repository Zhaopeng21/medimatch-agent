Day 03 —— Structured Conversation Memory
2026-07-30
为 MediMatch 增加结构化 Conversation Memory，优化多轮医疗问诊流程。
项目完成以下功能：
新增 PatientContext，用于保存结构化病情信息（症状、持续时间、严重程度等）
新增独立 Memory Node，实现病情信息提取与更新
将 Conversation Memory 接入 LangGraph Workflow
使用 Streamlit Session State 保存会话 Memory
优化药物推荐流程，使 RAG 检索基于结构化病情而非最后一句用户输入
保持原有分诊、Google Maps 查询及 UI 功能不受影响