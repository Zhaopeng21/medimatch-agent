Day 05 —— NZ 药物目录与项目收敛整理
2026-08-05

为 MediMatch 完成药物数据层迁移、项目目录整理与 Git 提交，减少旧印度品牌药资料对运行时推荐的影响，并明确后续架构收敛方向。

项目完成以下工作：

将运行时药物检索从旧的 Medicine_Details.csv 与 legacy FAISS 索引迁移至新的 NZ medicine catalog。新增 nz_medicine_catalog.json、metadata 与对应 FAISS 向量库，运行时仅加载新的药物目录。药物推荐输出改为通用有效成分与 Chemist Warehouse NZ 搜索入口，避免直接向用户展示旧资料中的印度品牌药。保留原始 CSV、PHARMAC Schedule、Medsafe 分类快照和成分候选表，用于数据来源追溯与后续重新构建 catalog。新增可重复执行的数据构建脚本，包括通用成分候选提取与 NZ catalog 构建流程。梳理 data/ 目录，将当前运行数据、pipeline 原始资料和 archive 历史备份分开管理。将旧向量库、历史库存资料和本地环境目录加入 .gitignore，避免不必要文件进入远程仓库。梳理当前 LangGraph 架构，明确后续将以 Triage 作为主医疗决策层，Tool Router 仅处理无症状的直接服务请求。完成代码、运行数据、数据构建流程与项目忽略规则的分批 Git 提交。成功将 4 个提交推送至 GitHub main 分支，工作区已恢复 clean 状态。

当前限制：

NZ catalog 目前覆盖的通用成分数量仍有限，部分轻症会因缺少可靠候选而安全回退。症状用途证据仍包含旧资料清洗结果，不应视为 NZ 临床指南。Triage 与 Memory 仍主要依赖 LLM，中文输入与信息完整性判断仍有优化空间。MediMatch 当前定位为 Auckland 医疗导航与技术演示项目，而非临床决策产品。