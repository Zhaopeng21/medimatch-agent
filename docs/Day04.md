Day 04 —— Tool Router 与安全优先级
2026-08-04

为 MediMatch 增加 Tool Router，将临床风险分诊与工具调用意图分离，优化多场景医疗咨询流程。

项目完成以下功能：
新增 ToolRoute，支持症状分诊、药物咨询、查找 GP、查找 Urgent Care 与一般医疗问题五类意图。
新增 Tool Router Node，根据用户当前问题和既有 PatientContext 判断应进入的功能路径。
重构 LangGraph Workflow：先执行 Conversation Memory 与 Triage，再由安全门判断是否存在紧急风险。
紧急症状优先进入 Urgent Care 路径，不会被药物、地点或一般问题查询绕过。
症状分诊继续复用原有分诊与药物推荐逻辑，保持现有功能稳定。
GP 与 Urgent Care 查询继续复用 Google Maps 工具。
药物咨询与一般医疗问题新增安全回退机制，避免在尚未完成专用药物 RAG 前提供不可靠或个体化的用药建议。
新增路由单元测试，覆盖紧急症状优先级、GP 查询、Urgent Care 查询、药物查询及症状补充等场景。
完成语法检查与单元测试验证，确保第一阶段 Tool Router 可稳定运行。