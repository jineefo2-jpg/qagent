// Detailed 12-week execution playbook for Agent learning
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak
} = require('docx');

const FONT = "Microsoft YaHei";

const border = { style: BorderStyle.SINGLE, size: 6, color: "B5C7D8" };
const allBorders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 100, bottom: 100, left: 140, right: 140 };

// ---------- helpers ----------
function tn(text, opts = {}) {
  return new TextRun({
    text, font: FONT, size: opts.size || 22,
    bold: opts.bold, color: opts.color, italics: opts.italics,
  });
}
function p(text, opts = {}) {
  const runs = Array.isArray(text) ? text : [tn(text, opts)];
  return new Paragraph({
    children: runs,
    spacing: { before: opts.before || 60, after: opts.after || 60, line: 320 },
    alignment: opts.align,
  });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [tn(text, { size: 36, bold: true, color: "1F4E79" })],
    spacing: { before: 360, after: 180 },
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [tn(text, { size: 28, bold: true, color: "2E75B6" })],
    spacing: { before: 280, after: 140 },
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [tn(text, { size: 24, bold: true, color: "385723" })],
    spacing: { before: 220, after: 100 },
  });
}
function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    children: [tn(text)],
    spacing: { before: 40, after: 40, line: 300 },
  });
}
function richBullet(runs, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    children: runs,
    spacing: { before: 40, after: 40, line: 300 },
  });
}
function callout(title, body, color = "FFF4CE", barColor = "F0C040") {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: 9360, type: WidthType.DXA },
      shading: { fill: color, type: ShadingType.CLEAR },
      margins: { top: 140, bottom: 140, left: 200, right: 200 },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 4, color: barColor },
        bottom: { style: BorderStyle.SINGLE, size: 4, color: barColor },
        left: { style: BorderStyle.SINGLE, size: 12, color: barColor },
        right: { style: BorderStyle.SINGLE, size: 4, color: barColor },
      },
      children: [
        new Paragraph({ children: [tn(title, { bold: true, size: 22, color: "5F4B00" })], spacing: { after: 60 } }),
        new Paragraph({ children: [tn(body, { color: "3F3000" })], spacing: { line: 300 } }),
      ],
    })]})],
  });
}
function makeTable(columnWidths, headerRow, rows) {
  const tableWidth = columnWidths.reduce((a, b) => a + b, 0);
  const headerCells = headerRow.map((text, i) =>
    new TableCell({
      width: { size: columnWidths[i], type: WidthType.DXA },
      shading: { fill: "1F4E79", type: ShadingType.CLEAR },
      borders: allBorders, margins: cellMargins,
      children: [new Paragraph({ children: [tn(text, { bold: true, color: "FFFFFF" })] })],
    })
  );
  const bodyRows = rows.map((row, ri) =>
    new TableRow({
      children: row.map((cell, ci) =>
        new TableCell({
          width: { size: columnWidths[ci], type: WidthType.DXA },
          shading: { fill: ri % 2 === 0 ? "F2F6FA" : "FFFFFF", type: ShadingType.CLEAR },
          borders: allBorders, margins: cellMargins,
          children: (Array.isArray(cell) ? cell : [cell]).map((line) =>
            new Paragraph({ children: [tn(String(line), { size: 21 })], spacing: { line: 280 } })
          ),
        })
      ),
    })
  );
  return new Table({
    width: { size: tableWidth, type: WidthType.DXA },
    columnWidths,
    rows: [new TableRow({ tableHeader: true, children: headerCells }), ...bodyRows],
  });
}

// Reusable week block
function weekBlock({ weekNo, theme, goal, days, papers, kbTasks, deliverables, selfCheck, extraNote }) {
  const out = [];
  out.push(h2(`第 ${weekNo} 周 · ${theme}`));
  out.push(callout("本周目标", goal, "EAF3FA", "2E75B6"));

  out.push(h3("7 天节奏（约 30 小时）"));
  out.push(makeTable(
    [900, 1600, 2900, 3960],
    ["日", "时段", "学习类型", "具体内容与产物"],
    days
  ));

  if (papers && papers.length) {
    out.push(h3("文献与课程清单（按优先级）"));
    out.push(makeTable(
      [800, 4400, 1400, 2760],
      ["优先级", "资料", "形式", "目的与读法"],
      papers
    ));
  }

  out.push(h3("知识库建设任务（写进个人 KB）"));
  for (const k of kbTasks) out.push(bullet(k));

  out.push(h3("本周交付物 / 验收"));
  for (const d of deliverables) out.push(bullet(d));

  out.push(h3("周末自测"));
  for (const q of selfCheck) out.push(bullet(q));

  if (extraNote) out.push(callout("提示", extraNote, "FFF4CE", "F0C040"));

  out.push(new Paragraph({ children: [new PageBreak()] }));
  return out;
}

const children = [];

// ---------- Cover ----------
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 600, after: 100 },
  children: [tn("Agent 学习执行手册", { size: 56, bold: true, color: "1F4E79" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 80 },
  children: [tn("12 周日级拆解 · 文献清单 · 知识库建设", { size: 30, color: "2E75B6" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 360 },
  children: [tn("配套《Agent 学习与训练路线》主文档使用", { size: 22, italics: true, color: "808080" })],
}));

// ---------- 0. How to use ----------
children.push(h1("0. 如何使用这份手册"));
children.push(p("这是配套主文档的执行版。主文档回答「学什么、做什么项目」，本手册回答「今天该读哪篇论文、做哪个练习、记哪张卡片」。建议按周打印或贴墙上，每天勾一项。"));

children.push(h2("0.1 三类学习内容的配比"));
children.push(makeTable(
  [1800, 1200, 6360],
  ["类型", "占比", "说明"],
  [
    ["原理文献", "约 25%", "论文 / 经典博客 / 课程，用于建立心智模型"],
    ["框架与工程", "约 30%", "官方文档 / 源码 / Cookbook，用于掌握工具"],
    ["动手实战", "约 35%", "本周代码 + 跑通 demo，用于把知识肌肉化"],
    ["知识库整理", "约 10%", "每周 2-3 小时把笔记重新组织进个人 KB"],
  ]
));

children.push(h2("0.2 文献阅读三遍法"));
children.push(richBullet([tn("第一遍（5 分钟）：", { bold: true }), tn("只读标题 / 摘要 / 图，决定值不值得读")]));
children.push(richBullet([tn("第二遍（30 分钟）：", { bold: true }), tn("通读但跳过推导，弄清楚问题、方法、结论")]));
children.push(richBullet([tn("第三遍（1-2 小时，仅核心论文）：", { bold: true }), tn("逐节精读，复现关键算法的伪代码或 demo")]));
children.push(callout("门槛",
  "标记为「核心」的文献必须做到第三遍。标记为「选读」的做到第二遍即可。不要让任何论文卡住一周——读不完先动手。",
  "FFF4CE", "F0C040"));

children.push(h2("0.3 个人知识库（KB）结构建议"));
children.push(p("推荐用 Notion / Obsidian / Logseq 任选其一，目录结构如下："));
children.push(bullet("00_Index：总入口，每周更新本周新增"));
children.push(bullet("10_Concepts：原子概念卡（Token、ReAct、RAG、Reflexion…）一卡一概念"));
children.push(bullet("20_Patterns：Agent 设计模式（Planner-Worker、Supervisor、Reflexion-Loop…）"));
children.push(bullet("30_Frameworks：框架笔记（LangChain、LangGraph、Agent SDK、MCP…）"));
children.push(bullet("40_Papers：论文卡片，按《问题/方法/亮点/局限/可借鉴》四段写"));
children.push(bullet("50_Code：代码片段、Prompt 模板、Tool 模板、Eval 模板"));
children.push(bullet("60_Domain_Quant：量化领域知识（策略、因子、指标、坑）"));
children.push(bullet("70_Projects：当前项目的设计文档与每周复盘"));
children.push(bullet("90_Inbox：临时收集，每周清理一次"));

children.push(h2("0.4 论文卡片模板"));
children.push(p("写到 40_Papers 下，每篇一张卡。模板："));
children.push(bullet("标题 / 作者 / 年份 / 链接 / 我的评分（1-5）"));
children.push(bullet("一句话总结（≤ 30 字）"));
children.push(bullet("解决什么问题（实际场景）"));
children.push(bullet("核心方法（≤ 200 字 + 一张图）"));
children.push(bullet("亮点（值得借鉴的 2-3 点）"));
children.push(bullet("局限（在我的项目里会不会成立）"));
children.push(bullet("代码 / 复现路径"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- Phase 1 ----------
children.push(h1("一、第一阶段（W1-W4）：原理打通"));

// W1
children.push(...weekBlock({
  weekNo: 1,
  theme: "LLM 工作原理 + Prompt 工程",
  goal: "把 LLM 的输入输出契约、Prompt 五大范式讲清楚；建立「LLM 是推理引擎」的心智模型。",
  days: [
    ["D1", "2-3h 概念", "原理", "读 Lilian Weng《LLM Powered Autonomous Agents》总览章节；KB 建第一张概念卡《LLM I/O Contract》"],
    ["D2", "2-3h 文献", "原理", "精读 Anthropic《Prompt Engineering Overview》+ 整理「Zero-shot/Few-shot/CoT/Self-Consistency/ReAct」5 张概念卡"],
    ["D3", "3-4h 动手", "实战", "用原生 API（不上框架）对同一任务跑 5 种 Prompt 范式，记录 token / 准确率 / 成本"],
    ["D4", "3h 动手", "实战", "学习 Structured Output：用 JSON Mode + Pydantic 让模型稳定输出结构化结果"],
    ["D5", "2h 文献 + 2h 动手", "结合", "读 OpenAI Cookbook「Reliable outputs」章节；改造昨天的 demo 加入 schema 校验与重试"],
    ["D6", "3h 进阶", "动手", "横向对比 Claude / GPT-4o / DeepSeek 三家在同一 Prompt 下的差异，KB 记录《模型选型笔记 v1》"],
    ["D7", "2-3h 复盘", "整理", "写一篇个人博客《我理解的 LLM I/O 契约》；KB 整理本周所有概念卡"],
  ],
  papers: [
    ["核心", "Lilian Weng《LLM Powered Autonomous Agents》", "博客", "全文精读，是后续所有 Agent 概念的总图"],
    ["核心", "Anthropic Prompt Engineering Guide（官方）", "文档", "通读，把每个技巧亲手试一遍"],
    ["核心", "DeepLearning.AI《ChatGPT Prompt Engineering for Developers》", "视频课", "1.5 倍速 1 天看完，重点是 system / user / assistant 角色"],
    ["选读", "Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022)", "论文", "三遍法第二遍即可，理解 CoT 的诱导机制"],
    ["选读", "OpenAI Cookbook：Techniques to improve reliability", "Cookbook", "做练习的同时随用随查"],
  ],
  kbTasks: [
    "10_Concepts：LLM I/O Contract / Token / Context Window / Temperature / Top-p / System Prompt",
    "10_Concepts：Zero-shot / Few-shot / CoT / Self-Consistency / ReAct（5 张）",
    "50_Code：5 个 Prompt 范式的可复用模板（Markdown 片段）",
    "30_Frameworks：原生 API 使用笔记（auth、retry、cost 估算）",
  ],
  deliverables: [
    "GitHub 仓库初始化，提交 W1 的 5 个 Prompt 范式对比 demo",
    "一篇 ≥ 1500 字的博客《我理解的 LLM I/O 契约》",
    "模型选型笔记 v1（Claude / GPT-4o / DeepSeek 对比表）",
  ],
  selfCheck: [
    "能在 5 分钟内对外讲清 ReAct 与 CoT 的区别？",
    "Temperature=0 真的是确定性的吗？为什么？",
    "Few-shot 的样例数量超过多少会反而变差？给出经验值与解释",
  ],
  extraNote: "第一周的产出量决定整条路线的节奏。如果 D3 之前还没在 GitHub push 任何代码，请先停下来调整作息。"
}));

// W2
children.push(...weekBlock({
  weekNo: 2,
  theme: "Tool Use + 手写 ReAct Loop",
  goal: "不依赖任何框架，用 200 行 Python 实现一个能调用工具的 Agent，理解循环本质。",
  days: [
    ["D1", "3h 文献", "原理", "精读 ReAct (Yao et al., 2022)，画出 Thought-Action-Observation 流程图"],
    ["D2", "2h 文献 + 2h 动手", "原理+实战", "读 Toolformer 论文（选读）+ Anthropic Tool Use 官方文档"],
    ["D3", "4h 动手", "实战", "用原生 API 实现 ReAct 主循环骨架（无工具版），跑通 Thought 输出"],
    ["D4", "4h 动手", "实战", "加入 3 个工具：calculator / web_search（Tavily 或 SerpAPI）/ calendar"],
    ["D5", "3h 进阶", "实战", "JSON Schema 设计、参数校验、工具调用失败的重试与降级策略"],
    ["D6", "3h 进阶", "实战", "故意制造异常：超时、模型幻觉、Schema 不符；写一份《容错策略备忘录》"],
    ["D7", "2-3h 复盘", "整理", "写博客《手写 ReAct：100 行能讲清楚的事》；KB 沉淀「ReAct 设计模式」卡片"],
  ],
  papers: [
    ["核心", "ReAct: Synergizing Reasoning and Acting in Language Models (Yao 2022)", "论文", "三遍法完整读完，必须能复述算法"],
    ["核心", "Anthropic Tool Use Documentation", "文档", "完整通读，重点看 parallel tool use 与错误处理"],
    ["选读", "Toolformer: Language Models Can Teach Themselves to Use Tools (Schick 2023)", "论文", "第二遍即可，理解 self-training 视角"],
    ["选读", "OpenAI Function Calling Cookbook", "文档", "对比 Anthropic 实现差异"],
  ],
  kbTasks: [
    "20_Patterns：ReAct Loop 设计模式卡（含伪代码）",
    "10_Concepts：Function Calling / JSON Schema / Tool Choice",
    "50_Code：极简 ReAct 模板（≤ 200 行可复用）",
    "50_Code：容错策略备忘录（超时、重试、降级、熔断）",
  ],
  deliverables: [
    "GitHub 提交 100-200 行的自建 Agent，README 包含运行示例",
    "5 个故障场景的容错测试用例 + 处理日志",
    "博客《手写 ReAct》",
  ],
  selfCheck: [
    "为什么 ReAct 比纯 CoT 更稳？写两个具体场景",
    "工具返回失败时，给出 3 种合理的恢复策略",
    "并行工具调用相比串行有什么风险？",
  ],
  extraNote: "这一周是「不偷懒」的关键。手写过的人和直接上 LangChain 的人，半年后的设计能力天差地别。"
}));

// W3
children.push(...weekBlock({
  weekNo: 3,
  theme: "LangChain 与 RAG 入门",
  goal: "理解 LangChain 的抽象层级；搭出第一个真正可用的 RAG 系统。",
  days: [
    ["D1", "3h 学习", "框架", "LangChain 官方 Tutorial：Runnable / LCEL / Chain，跑通 5 个示例"],
    ["D2", "3h 学习", "框架", "LangChain Agent / Tool / Memory 模块；对比 W2 自建版"],
    ["D3", "2h 文献 + 3h 动手", "结合", "读 Pinecone《Learn RAG》系列；搭建最朴素 RAG（10 篇 PDF）"],
    ["D4", "4h 进阶", "实战", "Chunking 策略对比：Fixed / Recursive / Semantic；Embedding 模型对比"],
    ["D5", "4h 进阶", "实战", "Hybrid Search（BM25 + 向量）+ Rerank（bge-reranker）"],
    ["D6", "3h 评估", "实战", "构建 20 题评估集；用 RAGAS 跑出 baseline 三项指标"],
    ["D7", "2-3h 复盘", "整理", "博客《我做的第一个 RAG 系统：从糟糕到可用》；KB 整理"],
  ],
  papers: [
    ["核心", "LangChain 官方文档 Tutorials（Quickstart + RAG + Agents）", "文档", "通读 + 跑示例"],
    ["核心", "Pinecone《Learn RAG》系列", "文章", "通读，建立 RAG 全流程心智图"],
    ["核心", "RAGAS 论文 & 文档", "论文+文档", "理解 faithfulness / answer_relevance / context_precision"],
    ["选读", "REALM / RETRO / Atlas 任选一篇", "论文", "了解 RAG 的学术演化"],
    ["选读", "BGE / E5 Embedding 模型论文", "论文", "理解向量检索的工程取舍"],
  ],
  kbTasks: [
    "30_Frameworks：LangChain 核心抽象（LCEL / Runnable / Agent）",
    "20_Patterns：RAG 标准流水线（Loader→Splitter→Embed→Store→Retrieve→Rerank→Generate）",
    "10_Concepts：Embedding / 余弦相似 / BM25 / Rerank / 评估指标",
    "50_Code：RAG 模板（含评估脚本）",
  ],
  deliverables: [
    "GitHub 提交研报问答 RAG，README 描述 chunking 与检索策略",
    "20 题评估集 + RAGAS 报告（baseline 数据）",
    "LangChain vs 自建 ReAct 的对比博客或表格",
  ],
  selfCheck: [
    "Chunk size 取 500 vs 1000 vs 2000 各有什么取舍？",
    "为什么需要 Rerank？说出它在准确率提升与延迟之间的取舍",
    "RAGAS 三个核心指标分别测什么？哪个最难提升？",
  ]
}));

// W4
children.push(...weekBlock({
  weekNo: 4,
  theme: "Claude Agent SDK + MCP + 三栈对比",
  goal: "理解 Agent SDK 的设计哲学与 MCP 协议；定型本路线后续主力栈。",
  days: [
    ["D1", "3h 学习", "框架", "Anthropic《Building Effective Agents》通读，记录 5 种 Agent 模式"],
    ["D2", "3h 学习", "框架", "Claude Agent SDK 官方文档 + Hooks / Permission / Subagent 设计"],
    ["D3", "3h 学习", "协议", "MCP 协议规范 + Server/Client 双视角；跑通官方示例 MCP Server"],
    ["D4", "4h 动手", "实战", "用 Agent SDK 复刻 W2 demo，重点体会 hooks / 工具自动发现"],
    ["D5", "4h 动手", "实战", "自己写一个 MCP Server 暴露 3 个工具，用 Claude Desktop 调用"],
    ["D6", "3h 整理", "对比", "完成三栈对比表（自建/LangChain/Agent SDK），选定主力栈并写选型理由"],
    ["D7", "2-3h 复盘", "整理", "博客《三种 Agent 实现方式对比》；KB 整理本阶段所有 Pattern"],
  ],
  papers: [
    ["核心", "Anthropic《Building Effective Agents》", "博客", "三遍法精读，是 Agent 工程的最佳入门指南"],
    ["核心", "Claude Agent SDK 官方文档", "文档", "通读 + 例子"],
    ["核心", "MCP（Model Context Protocol）规范", "规范文档", "理解为什么是「Tool 总线」"],
    ["选读", "Anthropic《How we built our multi-agent research system》", "博客", "看真实生产系统的设计取舍"],
  ],
  kbTasks: [
    "30_Frameworks：Claude Agent SDK 核心概念 + Hooks/Permission 笔记",
    "30_Frameworks：MCP 协议笔记 + 自建 Server 示例",
    "20_Patterns：5 种 Agent 模式（Anthropic 总结的：Prompt Chaining / Routing / Parallelization / Orchestrator-Workers / Evaluator-Optimizer）",
    "70_Projects：主力栈选型决策记录（ADR 风格）",
  ],
  deliverables: [
    "三栈对比表（含代码量、可调试、生态、成本）",
    "一个能被 Claude Desktop 调用的自建 MCP Server",
    "主力栈选型 ADR + 博客",
  ],
  selfCheck: [
    "Agent SDK 的 Permission System 解决了什么生产问题？",
    "MCP 相比直接函数调用，多了什么？少了什么？",
    "Anthropic 5 种 Agent 模式哪种最适合「研报问答」？哪种适合「量化研究」？",
  ],
  extraNote: "M1 里程碑：完成本周即达到主文档 M1。给自己一个奖励再进入第二阶段。"
}));

// ---------- Phase 2 ----------
children.push(h1("二、第二阶段（W5-W8）：进阶 + 量化领域 + MVP"));

// W5
children.push(...weekBlock({
  weekNo: 5,
  theme: "LangGraph 与多 Agent 协作",
  goal: "用状态机思维设计多 Agent 系统；跑通 Planner-Worker-Reviewer 三角结构。",
  days: [
    ["D1", "3h 学习", "框架", "LangGraph 官方 Quickstart + State / Node / Edge / Conditional 概念"],
    ["D2", "3h 学习", "框架", "LangGraph 进阶：Checkpoint / Interrupt / Human-in-the-loop"],
    ["D3", "2h 文献 + 2h 动手", "原理+实战", "读 MetaGPT 论文（选读）+ 跑通 LangGraph Supervisor 示例"],
    ["D4", "4h 动手", "实战", "自建 Planner-Worker-Reviewer：用户输入 → 拆解 → 并行执行 → 验收"],
    ["D5", "3h 进阶", "实战", "故意触发循环 / 状态污染，用 checkpoint + interrupt 调试"],
    ["D6", "3h 进阶", "实战", "实现 Routing 模式：根据问题类型自动选择不同子 Agent"],
    ["D7", "2-3h 复盘", "整理", "博客《多 Agent 设计：从循环到协作》；KB 整理多 Agent 模式卡"],
  ],
  papers: [
    ["核心", "LangGraph 官方 Tutorials（含 Supervisor / Multi-Agent）", "文档", "全部跑通"],
    ["核心", "Anthropic 多 Agent 文章 + AutoGen 设计文档", "博客", "对比两种学派的取舍"],
    ["选读", "MetaGPT (Hong et al., 2023)", "论文", "学习角色化 Agent 的 SOP 思想"],
    ["选读", "AutoGen (Wu et al., 2023)", "论文", "理解会话驱动的多 Agent"],
  ],
  kbTasks: [
    "20_Patterns：Supervisor / Hierarchical / Network / Plan-and-Execute 四种拓扑",
    "20_Patterns：Routing / Handoff 设计模式",
    "30_Frameworks：LangGraph State 设计经验、Checkpoint 使用",
    "50_Code：多 Agent 模板（Planner-Worker-Reviewer）",
  ],
  deliverables: [
    "GitHub 提交 Planner-Worker-Reviewer demo",
    "至少 3 个 bug 复现 + 调试过程记录（死循环/状态污染/上下文丢失）",
    "Routing demo（不同问题类型走不同分支）",
  ],
  selfCheck: [
    "Supervisor 模式与 Network 模式各自的适用场景？",
    "什么时候必须用 Checkpoint？什么时候可以省略？",
    "多 Agent 的「Context Bloat」如何缓解？",
  ]
}));

// W6
children.push(...weekBlock({
  weekNo: 6,
  theme: "RAG 工程化 + Memory 长期化",
  goal: "把 W3 的玩具 RAG 升级为生产级研报知识库；建立 Memory 分层模型。",
  days: [
    ["D1", "3h 学习", "原理", "Chunking 进阶：父子文档、表格特化、Hierarchical Summarization"],
    ["D2", "3h 学习", "原理", "Hybrid Search + Rerank 工程化、Query Rewriting / HyDE / Multi-Query"],
    ["D3", "4h 动手", "实战", "升级研报知识库：500 篇 PDF + Hybrid + Rerank + Citation"],
    ["D4", "3h 学习 + 2h 动手", "原理+实战", "Memory 分层：Working / Episodic / Semantic；接入 Mem0 或自建"],
    ["D5", "3h 进阶", "实战", "Adaptive RAG / Self-RAG / Corrective-RAG 任选一种实现"],
    ["D6", "3h 评估", "实战", "构建 50 题评估集，对比 baseline 与新版的 RAGAS 提升幅度"],
    ["D7", "2-3h 复盘", "整理", "博客《RAG 工程化的 10 个隐藏坑》；KB 整理 Memory 设计模式"],
  ],
  papers: [
    ["核心", "Self-RAG: Learning to Retrieve, Generate, and Critique (Asai 2023)", "论文", "三遍法读完，复现关键流程"],
    ["核心", "Corrective RAG (Yan et al., 2024)", "论文", "理解 Retrieval Evaluator 的设计"],
    ["核心", "HyDE: Hypothetical Document Embeddings (Gao 2022)", "论文", "理解查询改写思路"],
    ["选读", "Lost in the Middle (Liu et al., 2023)", "论文", "长上下文召回顺序的实证研究"],
    ["选读", "Mem0 / MemGPT 论文与博客", "混合", "理解长期记忆的工程化"],
  ],
  kbTasks: [
    "20_Patterns：高级 RAG 模式（Self-RAG / Corrective-RAG / Adaptive-RAG）",
    "20_Patterns：Memory 三层模型卡片",
    "10_Concepts：Query Rewriting / HyDE / Multi-Query / Cross-Encoder Rerank",
    "50_Code：研报 RAG v2 模板（带 citation）",
  ],
  deliverables: [
    "研报知识库 v2 上线，支持 500+ 文档，引用回溯",
    "RAGAS 评估对比：v1 vs v2 提升幅度图",
    "Memory 分层 demo（对话历史 + 用户偏好 + 知识库）",
  ],
  selfCheck: [
    "为什么单纯加大 chunk 通常不会提升 RAG 质量？",
    "Self-RAG 与 Corrective-RAG 的关键差异？",
    "Memory 分层模型中，哪些必须实时写，哪些可异步？",
  ]
}));

// W7
children.push(...weekBlock({
  weekNo: 7,
  theme: "量化投资知识 + 工具链准备",
  goal: "在不写 LLM 代码的前提下，把量化研究的工程基础全部跑通。",
  days: [
    ["D1", "3h 学习", "领域", "市场结构、交易规则、数据频率；A 股 vs 美股差异（《量化投资策略》前 3 章）"],
    ["D2", "3h 学习", "领域", "策略类型与评价指标（夏普 / 最大回撤 / 卡玛 / 信息比率）"],
    ["D3", "3h 学习 + 2h 动手", "工程", "AKShare 安装与基础用法；拉取沪深 300 历史日线存入 DuckDB"],
    ["D4", "4h 动手", "工程", "Backtrader 上手：跑通官方双均线 demo + 加上交易成本"],
    ["D5", "4h 动手", "工程", "用 Backtrader 自己写 3 种策略：双均线 / RSI 反转 / 布林带"],
    ["D6", "3h 进阶", "工程", "策略报告自动化：Pyfolio / Quantstats 生成 HTML 报告"],
    ["D7", "2-3h 复盘", "整理", "博客《量化研究的 5 个工程化要点》；KB 建领域子目录"],
  ],
  papers: [
    ["核心", "《主动投资组合管理》Grinold & Kahn 第 1-4 章", "书", "理解 Alpha、信息比率、IC、IR"],
    ["核心", "Alpha 101 (Kakushadze, 2015)", "论文", "读懂表达式语法，能复现 5 个因子"],
    ["核心", "Backtrader 官方文档 Quickstart + Strategy", "文档", "通读 + 跑通"],
    ["选读", "Worldquant Brain 公开因子库", "网站", "浏览，挑选 10 个感兴趣的"],
    ["选读", "AQR 公开研究：Value & Momentum Everywhere", "论文", "了解学术派经典策略"],
  ],
  kbTasks: [
    "60_Domain_Quant：市场结构、交易规则、数据频率",
    "60_Domain_Quant：策略类型分类卡 + 评价指标卡（夏普/回撤/IC/IR）",
    "60_Domain_Quant：典型坑（未来函数、幸存者偏差、过拟合、滑点）",
    "50_Code：数据管道模板 + Backtrader 策略模板",
  ],
  deliverables: [
    "数据管道：AKShare → DuckDB，至少 5 年沪深 300 日线",
    "3 个可运行 Backtrader 策略 + 标准化回测报告",
    "策略指标计算工具函数库（≥ 8 个指标）",
  ],
  selfCheck: [
    "未来函数最常见的 3 种来源？怎么自动检测？",
    "夏普比率为什么有时不能反映真实风险？",
    "回测中加入交易成本对结果可能产生几个百分点的影响？",
  ],
  extraNote: "本周不写 Agent 代码。让自己暂时退回「纯量化研究员」视角，否则后面 LLM 写代码会缺乏判断力。"
}));

// W8
children.push(...weekBlock({
  weekNo: 8,
  theme: "量化 Agent MVP：自然语言到回测报告",
  goal: "完成 M2 里程碑：用户一句话 → Agent 自动产出回测报告。",
  days: [
    ["D1", "2h 设计", "工程", "MVP 设计文档：Tool 列表 / State 设计 / Prompt 草稿"],
    ["D2", "5h 动手", "工程", "实现 fetch_market_data + generate_strategy_code 两个 Tool"],
    ["D3", "5h 动手", "工程", "实现 run_backtest（沙箱执行）+ 报告生成 Tool"],
    ["D4", "4h 动手", "工程", "用 LangGraph 把 4 个 Tool 编排为闭环，Streamlit 前端"],
    ["D5", "3h 进阶", "实战", "加入 critique_strategy（自我反思）；跑 5 个典型 query 调试"],
    ["D6", "3h 评估", "实战", "构建 MVP 评估集：10 个用户 query + 期望结果"],
    ["D7", "2-3h 复盘", "整理", "录 5 分钟 demo 视频；博客《我的第一个量化 Agent》"],
  ],
  papers: [
    ["核心", "Anthropic Code Execution Tool 文档", "文档", "理解沙箱执行模式"],
    ["核心", "FinGPT / FinRobot 项目 README + 关键设计", "代码", "借鉴金融领域的 prompt 与工具设计"],
    ["选读", "AutoGen Code Execution 示例", "代码", "对比另一种执行模式"],
  ],
  kbTasks: [
    "70_Projects：量化 Agent v1 设计文档（架构 / Tool / Prompt / State）",
    "50_Code：sandbox 执行模板（Docker / restrictedpython）",
    "50_Code：Strategy DSL 草案（用 Pydantic 定义策略规范）",
    "60_Domain_Quant：MVP 评估集（10 个 query）",
  ],
  deliverables: [
    "MVP 可演示版本：3 类策略 90 秒内出报告",
    "Streamlit 前端可对话",
    "5 分钟演示视频 + 设计博客",
  ],
  selfCheck: [
    "如果用户说「写一个能赚钱的策略」，Agent 应该怎么拒绝并引导？",
    "策略代码生成失败时，Agent 如何自我修复？",
    "如何防止 LLM 在策略代码里偷偷使用未来函数？",
  ],
  extraNote: "M2 里程碑：完成本周即达到主文档 M2。中场休息一天再进入第三阶段。"
}));

// ---------- Phase 3 ----------
children.push(h1("三、第三阶段（W9-W12）：项目深化与产品化"));

// W9
children.push(...weekBlock({
  weekNo: 9,
  theme: "因子挖掘 Agent",
  goal: "Agent 能基于市场假设自动生成、评估、筛选因子。",
  days: [
    ["D1", "3h 学习", "领域+原理", "Alpha 101 表达式语法；FunSearch / Symbolic Regression 思路"],
    ["D2", "2h 文献", "原理", "读 Voyager 论文（核心）+ FunSearch 博客，理解「自动生成代码」范式"],
    ["D3", "4h 动手", "实战", "设计 Factor DSL（rank、ts_mean、correlation 等基础算子）"],
    ["D4", "4h 动手", "实战", "实现 generate_factor_expr → evaluate_factor(IC/IR) Tool"],
    ["D5", "4h 动手", "实战", "加入 Self-Reflection：Agent 看完结果迭代下一个候选"],
    ["D6", "3h 评估", "实战", "跑 20 个市场假设，每个生成 10 候选因子，输出 Top-N 报告"],
    ["D7", "2-3h 复盘", "整理", "博客《让 LLM 当因子工程师：能做到什么、做不到什么》"],
  ],
  papers: [
    ["核心", "Voyager: An Open-Ended Embodied Agent with LLMs (Wang 2023)", "论文", "三遍法精读，借鉴 skill library 思路"],
    ["核心", "FunSearch (DeepMind, 2023)", "博客+论文", "理解 LLM 做符号搜索的范式"],
    ["核心", "Alpha 101 论文", "论文", "重温因子表达式语法"],
    ["选读", "AlphaGen / AlphaForge 等学术尝试", "论文", "看研究界如何用神经网络/RL 做因子"],
  ],
  kbTasks: [
    "20_Patterns：Skill Library / 自迭代生成 设计模式",
    "60_Domain_Quant：Factor DSL 设计文档",
    "60_Domain_Quant：因子评估指标卡（IC / IR / 分层收益 / 换手率）",
    "70_Projects：因子挖掘 Agent 实验日志",
  ],
  deliverables: [
    "Factor DSL + 10 个基础算子可运行",
    "20 个假设 × 10 因子的实验报告",
    "Self-Reflection 的前后对比数据",
  ],
  selfCheck: [
    "为什么不让 LLM 直接写 numpy 代码而要走 DSL？",
    "因子的 IC 高 ≠ 实盘赚钱，原因是什么？",
    "Self-Reflection 在你的实验里把效果提升了多少？为什么？",
  ]
}));

// W10
children.push(...weekBlock({
  weekNo: 10,
  theme: "回测 Agent + 评估循环",
  goal: "回测端引入 Reflexion 范式，自动诊断与改进；建立专家规则集。",
  days: [
    ["D1", "3h 文献", "原理", "精读 Reflexion + Self-Refine 论文"],
    ["D2", "3h 文献", "原理", "Critique / Verifier 模式综述（Eugene Yan 博客）"],
    ["D3", "4h 动手", "实战", "实现 Actor-Evaluator-Reflector 三层结构"],
    ["D4", "4h 动手", "实战", "编码专家规则集：未来函数检测、过拟合检测、稳健性检查（≥ 10 条）"],
    ["D5", "3h 进阶", "实战", "成本路由：编码用 Claude，批量评估用 DeepSeek，简单分类用本地模型"],
    ["D6", "3h 评估", "实战", "对比有/无 Reflexion 的最终回测质量与稳健性"],
    ["D7", "2-3h 复盘", "整理", "博客《让 Agent 学会怀疑自己：回测中的 Reflexion 实战》"],
  ],
  papers: [
    ["核心", "Reflexion: Language Agents with Verbal RL (Shinn 2023)", "论文", "三遍法精读"],
    ["核心", "Self-Refine: Iterative Refinement with Self-Feedback (Madaan 2023)", "论文", "三遍法精读"],
    ["核心", "Eugene Yan《Patterns for Building LLM-based Systems》", "博客", "重点看 Eval / Guardrails 章节"],
    ["选读", "Constitutional AI (Anthropic, 2022)", "论文", "理解规则化反思的极端版本"],
  ],
  kbTasks: [
    "20_Patterns：Actor-Evaluator-Reflector 设计模式",
    "20_Patterns：成本路由策略卡（按任务类型选模型）",
    "60_Domain_Quant：专家规则集（≥ 10 条）",
    "70_Projects：Reflexion 前后对比实验",
  ],
  deliverables: [
    "Reflexion 闭环代码 + 实验数据",
    "10+ 专家规则的可执行检查器",
    "成本对比：Reflexion 前后 token 与单次任务费用",
  ],
  selfCheck: [
    "Reflexion 在什么场景下会反而变差？",
    "专家规则与 LLM 评估各自的优劣？怎么混用？",
    "如何避免 Reflector 与 Actor 形成「互相吹捧」的局面？",
  ]
}));

// W11
children.push(...weekBlock({
  weekNo: 11,
  theme: "多 Agent 编排：仿真量化研究院",
  goal: "把所有能力组装成 6 角色多 Agent 系统；可一键完成完整研究流程。",
  days: [
    ["D1", "2h 设计", "工程", "总体架构设计：Supervisor + 5 个角色 Agent + 共享 State"],
    ["D2", "4h 动手", "工程", "实现 Supervisor + 资讯分析师 + 因子工程师两个角色"],
    ["D3", "4h 动手", "工程", "实现 策略工程师 + 回测工程师 + 风控审查员"],
    ["D4", "4h 动手", "工程", "Checkpoint + 中断恢复 + 人工介入入口"],
    ["D5", "3h 进阶", "实战", "可观测性：每个 Agent 的调用次数 / 平均耗时 / token 成本 dashboard"],
    ["D6", "3h 评估", "实战", "跑 20 个端到端任务，统计成功率与平均成本"],
    ["D7", "2-3h 复盘", "整理", "博客《我的 6 角色量化研究院：成功率 / 成本 / 局限》"],
  ],
  papers: [
    ["核心", "Anthropic《How we built our multi-agent research system》", "博客", "再次精读，对照自己的系统"],
    ["核心", "Cognition《Don't Build Multi-Agents》", "博客", "理解反方观点，对自己的设计提质疑"],
    ["核心", "AutoGen / CrewAI / MetaGPT 三家代表项目对比博客", "博客", "看不同学派的工程取舍"],
    ["选读", "ChatDev (Qian et al., 2023)", "论文", "另一个角色化多 Agent 系统"],
  ],
  kbTasks: [
    "70_Projects：6 角色研究院架构总图（要能贴墙上）",
    "20_Patterns：Observability 设计模式（trace / metric / log）",
    "50_Code：可复用的 Supervisor 模板",
    "70_Projects：20 个端到端任务的实验记录",
  ],
  deliverables: [
    "6 角色完整可运行系统",
    "Observability dashboard（哪怕只是个简单 Streamlit）",
    "端到端成功率 ≥ 70% 的证据",
  ],
  selfCheck: [
    "如果只能保留 3 个角色 Agent，你会保留哪三个？为什么？",
    "Cognition 反对多 Agent 的论点中，你认同与不认同的部分？",
    "你的系统目前最大的成本来源是？最大的延迟来源是？",
  ]
}));

// W12
children.push(...weekBlock({
  weekNo: 12,
  theme: "产品化、评估、上线、复盘",
  goal: "完成 M4 里程碑：演示视频 / 完整复盘 / 经验沉淀。",
  days: [
    ["D1", "3h 工程", "产品", "前端打磨：对话历史、文件上传、报告导出"],
    ["D2", "3h 工程", "产品", "私有部署考虑：API key 管理、限流、本地化数据"],
    ["D3", "3h 评估", "工程", "完整评估脚本：跑预置 20 个任务并生成评估报告"],
    ["D4", "3h 工程", "产品", "可观测性面板正式版（Langfuse 或自建）"],
    ["D5", "3h 输出", "整理", "录制 5 分钟产品演示视频"],
    ["D6", "4h 输出", "整理", "完成 5000 字技术博客 / 项目复盘"],
    ["D7", "2-3h 复盘", "整理", "总复盘：12 周个人能力对照表 + 下一阶段路线"],
  ],
  papers: [
    ["核心", "Anthropic《Engineering / Production》系列博客", "博客", "上线前的最后一遍 checklist"],
    ["核心", "Eugene Yan《Eval Driven Development》", "博客", "评估驱动思维的总结"],
    ["选读", "OpenAI Evals 框架", "代码", "看大厂如何工程化评估"],
  ],
  kbTasks: [
    "70_Projects：完整复盘文档（学到的 / 走过的弯路 / 下一步）",
    "20_Patterns：本路线总结的 10 大设计模式卡片",
    "00_Index：更新总入口，串联整套 KB",
    "50_Code：把可复用模板独立成 GitHub starter 模板",
  ],
  deliverables: [
    "演示视频（5 分钟）",
    "5000 字技术博客 / 复盘",
    "GitHub starter 模板（其他人可一键 fork）",
    "下一阶段 3-6 个月的延伸学习计划草案",
  ],
  selfCheck: [
    "12 周自测题 10 道，回头答能答 ≥ 8 道？",
    "如果让你重做一次，会先做不同的 3 件事是？",
    "你的项目对真实量化研究员有用吗？请引用 2 个具体反馈",
  ],
  extraNote: "完成本周即达到主文档 M4 全部里程碑。继续 W13+ 路线见下一章。"
}));

// ---------- 4. After 12 weeks ----------
children.push(h1("四、12 周之后：延伸路线"));

children.push(h2("4.1 第 13-16 周：技术深化"));
children.push(bullet("精读源码：LangGraph、Claude Agent SDK、AutoGen 任选其一"));
children.push(bullet("Agent Eval 进阶：建立可复用的评估流水线（Promptfoo / DeepEval）"));
children.push(bullet("一篇高质量技术博客，公开发表（公众号 / Medium / 个人博客）"));

children.push(h2("4.2 第 17-20 周：领域迁移"));
children.push(bullet("挑一个新领域（法律 / 医疗 / 研发 / 客服），用 4 周做 MVP"));
children.push(bullet("把量化项目里沉淀的 Pattern、模板、Evaluator 复用过去"));
children.push(bullet("对比文档：两个领域 Agent 设计的「不变量」与「领域差异」"));

children.push(h2("4.3 第 21-24 周：影响力构建"));
children.push(bullet("把 GitHub starter 模板推到 500+ star（持续维护、写示例）"));
children.push(bullet("做 1 次线下分享或线上技术 talk"));
children.push(bullet("尝试一个垂直产品的小范围用户验证"));

children.push(h2("4.4 长期 4 类能力刻意练习"));
children.push(makeTable(
  [2400, 6960],
  ["能力", "练习方式"],
  [
    ["架构判断力", "看新框架时先猜其设计取舍，再读官方解释，对比差异"],
    ["原理表达力", "每月 1 篇博客，每月 1 次对外讲解（哪怕只是给同事）"],
    ["工程稳健性", "每个项目都加上：评估集、可观测、成本监控、错误兜底"],
    ["领域迁移力", "至少在 3 个不同垂直领域各完成 1 个 MVP，沉淀 1 套通用模板"],
  ]
));

// ---------- 5. Reading & Knowledge appendix ----------
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("五、附录：文献全清单与知识库索引"));

children.push(h2("5.1 必读论文（按阅读时机排序）"));
children.push(makeTable(
  [900, 4800, 1300, 2360],
  ["读期", "论文", "类别", "理由"],
  [
    ["W1", "Lilian Weng《LLM Powered Autonomous Agents》", "总览", "全图入门"],
    ["W2", "ReAct (Yao 2022)", "Agent 基础", "理解循环本质"],
    ["W2", "Toolformer (Schick 2023)", "Tool Use", "自训练视角（选读）"],
    ["W3", "Self-RAG (Asai 2023)", "RAG 进阶", "可控检索范式"],
    ["W3", "Corrective RAG (Yan 2024)", "RAG 进阶", "评估器思想"],
    ["W3", "HyDE (Gao 2022)", "查询改写", "提升召回"],
    ["W4", "Anthropic《Building Effective Agents》", "工程", "5 大 Agent 模式"],
    ["W4", "Anthropic《How we built our multi-agent...》", "工程", "真实生产案例"],
    ["W5", "MetaGPT (Hong 2023)", "多 Agent", "角色化 SOP（选读）"],
    ["W5", "AutoGen (Wu 2023)", "多 Agent", "会话驱动（选读）"],
    ["W6", "Lost in the Middle (Liu 2023)", "长上下文", "RAG 长上下文实证"],
    ["W7", "Alpha 101 (Kakushadze 2015)", "量化", "因子表达式范本"],
    ["W9", "Voyager (Wang 2023)", "Agent 进阶", "Skill Library 思想"],
    ["W9", "FunSearch (DeepMind 2023)", "代码生成", "符号搜索范式"],
    ["W10", "Reflexion (Shinn 2023)", "自我反思", "Verbal RL"],
    ["W10", "Self-Refine (Madaan 2023)", "自我反思", "迭代精炼"],
    ["W11", "ChatDev (Qian 2023)", "多 Agent", "角色化研发流程（选读）"],
  ]
));

children.push(h2("5.2 必读博客 / 课程（持续翻阅）"));
children.push(bullet("Lilian Weng（OpenAI）个人博客：Agent / RL / Diffusion 三大主题"));
children.push(bullet("Eugene Yan：《Patterns for Building LLM-based Systems》《Eval-Driven Development》"));
children.push(bullet("Anthropic Engineering Blog：尤其是 Effective Agents / Multi-Agent / Tool Use"));
children.push(bullet("Hamel Husain：《Your AI Product Needs Evals》"));
children.push(bullet("Chip Huyen：MLOps + LLMOps 文章"));
children.push(bullet("DeepLearning.AI 短课系列：LangChain / Agents / RAG / Evals"));

children.push(h2("5.3 知识库目录最终形态（W12 时应长这样）"));
children.push(p("到 W12 结束，你的个人 KB 应该至少包含："));
children.push(bullet("10_Concepts：≥ 30 张原子概念卡"));
children.push(bullet("20_Patterns：≥ 12 张设计模式卡（ReAct / Reflexion / Supervisor / RAG 流水线…）"));
children.push(bullet("30_Frameworks：4 套框架笔记（原生 / LangChain / LangGraph / Agent SDK）"));
children.push(bullet("40_Papers：≥ 15 张论文卡"));
children.push(bullet("50_Code：≥ 10 个可复用模板（Prompt / Tool / Eval / Sandbox / RAG …）"));
children.push(bullet("60_Domain_Quant：≥ 15 张量化领域卡"));
children.push(bullet("70_Projects：3 份正式设计文档 + 12 份周复盘"));

children.push(h2("5.4 每周节奏小提醒"));
children.push(callout(
  "三条铁律",
  "1) 周末必须有 GitHub commit；2) 每周必须新增 ≥ 5 张 KB 卡片；3) 每两周必须有 1 篇博客或长笔记对外可分享。坚持 12 周，能力会肉眼可见地复利增长。",
  "EAF3FA", "2E75B6"
));

// ---------- Build ----------
const doc = new Document({
  creator: "Claude",
  title: "Agent 学习执行手册",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: FONT, color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: "2E75B6" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "385723" },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
      ],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [tn("Agent 学习执行手册 · 12 周日级拆解", { size: 18, color: "808080", italics: true })],
      })]}),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          tn("第 ", { size: 18, color: "808080" }),
          new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "808080" }),
          tn(" 页 · 共 ", { size: 18, color: "808080" }),
          new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 18, color: "808080" }),
          tn(" 页", { size: 18, color: "808080" }),
        ],
      })]}),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = "/sessions/loving-elegant-archimedes/mnt/outputs/Agent学习执行手册_12周日级拆解.docx";
  fs.writeFileSync(out, buffer);
  console.log("Wrote:", out, "size:", buffer.length);
});
