// Build a comprehensive Agent learning & training plan document
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, TabStopType, TabStopPosition,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber,
  PageBreak, TableOfContents, PageOrientation
} = require('docx');

// ---------- Helpers ----------
const FONT_CN = "Microsoft YaHei"; // safe Chinese-capable font; falls back gracefully
const FONT_EN = "Arial";

const border = { style: BorderStyle.SINGLE, size: 6, color: "B5C7D8" };
const allBorders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 100, bottom: 100, left: 140, right: 140 };

function p(text, opts = {}) {
  const runs = Array.isArray(text)
    ? text
    : [new TextRun({ text, font: FONT_CN, size: opts.size || 22, bold: opts.bold, color: opts.color, italics: opts.italics })];
  return new Paragraph({
    children: runs,
    spacing: { before: opts.before || 60, after: opts.after || 60, line: 320 },
    alignment: opts.align,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: FONT_CN, size: 36, bold: true, color: "1F4E79" })],
    spacing: { before: 360, after: 180 },
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: FONT_CN, size: 28, bold: true, color: "2E75B6" })],
    spacing: { before: 280, after: 140 },
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: FONT_CN, size: 24, bold: true, color: "385723" })],
    spacing: { before: 220, after: 100 },
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    children: [new TextRun({ text, font: FONT_CN, size: 22 })],
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

function tn(text, opts = {}) {
  return new TextRun({
    text,
    font: FONT_CN,
    size: opts.size || 22,
    bold: opts.bold,
    color: opts.color,
    italics: opts.italics,
  });
}

function callout(title, body, color = "FFF4CE") {
  // A single-cell shaded paragraph block used as a callout (no borders ⇒ looks like a soft block)
  const titlePara = new Paragraph({
    children: [new TextRun({ text: title, font: FONT_CN, bold: true, size: 22, color: "5F4B00" })],
    spacing: { before: 40, after: 60 },
  });
  const bodyPara = new Paragraph({
    children: [new TextRun({ text: body, font: FONT_CN, size: 22, color: "3F3000" })],
    spacing: { before: 0, after: 40, line: 300 },
  });
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: 9360, type: WidthType.DXA },
            shading: { fill: color, type: ShadingType.CLEAR },
            margins: { top: 140, bottom: 140, left: 200, right: 200 },
            borders: {
              top: { style: BorderStyle.SINGLE, size: 4, color: "F0C040" },
              bottom: { style: BorderStyle.SINGLE, size: 4, color: "F0C040" },
              left: { style: BorderStyle.SINGLE, size: 12, color: "F0C040" },
              right: { style: BorderStyle.SINGLE, size: 4, color: "F0C040" },
            },
            children: [titlePara, bodyPara],
          }),
        ],
      }),
    ],
  });
}

function makeTable(columnWidths, headerRow, rows) {
  const tableWidth = columnWidths.reduce((a, b) => a + b, 0);
  const headerCells = headerRow.map((text, i) =>
    new TableCell({
      width: { size: columnWidths[i], type: WidthType.DXA },
      shading: { fill: "1F4E79", type: ShadingType.CLEAR },
      borders: allBorders,
      margins: cellMargins,
      children: [
        new Paragraph({
          children: [new TextRun({ text, font: FONT_CN, size: 22, bold: true, color: "FFFFFF" })],
        }),
      ],
    })
  );
  const bodyRows = rows.map((row, ri) =>
    new TableRow({
      children: row.map((cell, ci) =>
        new TableCell({
          width: { size: columnWidths[ci], type: WidthType.DXA },
          shading: { fill: ri % 2 === 0 ? "F2F6FA" : "FFFFFF", type: ShadingType.CLEAR },
          borders: allBorders,
          margins: cellMargins,
          children: (Array.isArray(cell) ? cell : [cell]).map((line) =>
            new Paragraph({
              children: [new TextRun({ text: String(line), font: FONT_CN, size: 21 })],
              spacing: { line: 280 },
            })
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

// ---------- Document content ----------

const children = [];

// Cover-ish title block
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 600, after: 120 },
  children: [new TextRun({ text: "Agent 底层学习与应用开发", font: FONT_CN, size: 56, bold: true, color: "1F4E79" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 80 },
  children: [new TextRun({ text: "3 个月密集学习 × 量化投资 Agent 实战路线", font: FONT_CN, size: 32, color: "2E75B6" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 360 },
  children: [new TextRun({ text: "学习计划 · 训练计划 · 项目蓝图 · 资源清单", font: FONT_CN, size: 22, italics: true, color: "808080" })],
}));

// Quick meta table
children.push(makeTable(
  [2200, 7160],
  ["项目", "说明"],
  [
    ["学习者画像", "Python 基础扎实，了解 LLM 基本使用，目标成为资深 AI 研发"],
    ["学习周期", "3 个月密集学习（每周 25-35 小时投入），共 12 周"],
    ["技术栈", "LangChain / LangGraph + Claude Agent SDK + 原生 API + 自建框架（对比学习）"],
    ["旗舰项目", "专业量化投资 Agent：策略生成 + 因子挖掘 + 自动回测 + 多 Agent 协作"],
    ["延伸能力", "项目方法论可平移到任何垂直领域（法律、医疗、客服、研发等）"],
    ["最终产出", "1 套个人方法论 + 1 个可演示量化 Agent 产品 + 1 份技术博客/复盘"],
  ]
));
children.push(p(""));

// Vision / Why this plan
children.push(h1("一、整体目标与方法论"));
children.push(p("本路线服务两个高层目标：第一，把 Agent 的原理彻底打通——不仅会用，更知道为什么这样设计；第二，用一个高难度的垂直项目（量化投资 Agent）作为试金石，把学到的每一块能力都落到工程上。"));
children.push(p("方法论遵循三条原则："));
children.push(richBullet([tn("原理先行：", { bold: true }), tn("先把 LLM、Tool Use、ReAct、Planning、Memory、Multi-Agent 这些核心概念读到能给别人讲清楚的程度，再开始抄代码；")]));
children.push(richBullet([tn("以战代练：", { bold: true }), tn("每周必须有可运行的产物（代码、demo、博客），不允许只看不写；")]));
children.push(richBullet([tn("框架对比：", { bold: true }), tn("同一个能力点用 LangChain / Agent SDK / 原生 API 各实现一次，对比其抽象差异，最终选定主力栈。")]));
children.push(callout("贯穿整条路线的一句话",
  "Agent 的本质是「LLM 作为推理引擎，调用工具与记忆，按目标循环执行」。一切框架都是这个循环的工程化包装。理解循环，你就能在任何框架上自由切换。"));
children.push(p(""));

// Roadmap overview
children.push(h1("二、12 周学习路线图（鸟瞰）"));
children.push(makeTable(
  [900, 2200, 3260, 3000],
  ["周次", "阶段", "核心主题", "里程碑产出"],
  [
    ["W1", "基础打通", "LLM 工作原理 / Prompt 工程 / Tool Use 协议", "1 篇笔记 + 5 个 Prompt 范式 demo"],
    ["W2", "基础打通", "ReAct / Function Calling / Structured Output", "第一个能调用工具的迷你 Agent"],
    ["W3", "框架入门", "LangChain 核心：Chain / Agent / Tool / Memory", "用 LangChain 复刻 W2 Agent + 增加 RAG"],
    ["W4", "框架入门", "Claude Agent SDK + 原生 API 自建 Loop", "三种实现方式横向对比表"],
    ["W5", "进阶能力", "LangGraph 状态机 / 多 Agent 协作 / Routing", "一个 Planner-Worker 多 Agent demo"],
    ["W6", "进阶能力", "RAG 工程化 / 向量库 / Memory 长期化", "金融研报知识库 v1"],
    ["W7", "领域准备", "量化投资知识、数据源、回测框架", "数据管道 + Backtrader 跑通示例"],
    ["W8", "项目 MVP", "量化 Agent MVP：自然语言→策略代码", "可输入「双均线」生成可运行策略"],
    ["W9", "项目深化", "因子挖掘 Agent + Code Interpreter", "自动生成 10 个候选因子并评估"],
    ["W10", "项目深化", "回测 Agent + 自我反思 / 评估循环", "策略自动跑回测并产出可视化报告"],
    ["W11", "系统化", "多 Agent 编排：研究员-分析师-回测员-风控", "完整工作流可一键运行"],
    ["W12", "产品化", "评估、监控、上线、写复盘", "产品 Demo + 技术博客 + 经验沉淀"],
  ]
));
children.push(p(""));

children.push(new Paragraph({ children: [new PageBreak()] }));

// Phase 1
children.push(h1("三、第一阶段（W1-W4）：原理与基础工具"));
children.push(p("目标：把 Agent 的核心概念全部串通，并能用三种方式实现「能调用工具」的 Agent。"));

children.push(h2("W1 · LLM 与 Prompt 工程基础"));
children.push(h3("学习内容"));
children.push(bullet("LLM 的核心机制：Token / Context Window / Temperature / Sampling，能用一张图讲清楚"));
children.push(bullet("Prompt 工程的五大范式：Zero-shot / Few-shot / CoT / Self-Consistency / ReAct"));
children.push(bullet("Claude、GPT、国产模型的能力差异与价格对比"));
children.push(bullet("System Prompt 设计、角色设定、输出约束（JSON Mode / Structured Output）"));
children.push(h3("动手任务"));
children.push(bullet("用 5 种 Prompt 范式分别解决「研报摘要 + 风险点提取」同一任务，量化对比效果"));
children.push(bullet("写一篇笔记《我理解的 LLM 输入输出契约》，作为后续所有 Agent 设计的理论基石"));
children.push(h3("推荐资源"));
children.push(bullet("Anthropic Prompt Engineering Guide（官方文档）"));
children.push(bullet("OpenAI Cookbook 中 prompting 章节"));
children.push(bullet("吴恩达《ChatGPT Prompt Engineering for Developers》课程"));

children.push(h2("W2 · Tool Use 与第一个 Agent"));
children.push(h3("学习内容"));
children.push(bullet("Function Calling 协议：JSON Schema、参数校验、错误处理"));
children.push(bullet("ReAct 循环：Thought → Action → Observation → ...，手写一遍最朴素的 while 循环"));
children.push(bullet("Tool 的设计原则：单一职责、可观测、错误自愈"));
children.push(bullet("Structured Output：用 Pydantic / Zod 约束 LLM 返回"));
children.push(h3("动手任务"));
children.push(bullet("用原生 API（不上框架）实现一个 Agent：能调用「Web 搜索 + 计算器 + 日历」三个工具"));
children.push(bullet("故意触发各种异常（工具 timeout、JSON 解析失败、模型胡说），写出容错策略"));
children.push(callout("关键认知",
  "不要急着上 LangChain。亲手写一遍 ReAct loop 是这条路线最重要的一次「不要偷懒」时刻——后面所有框架的设计取舍你都能秒懂。"));

children.push(h2("W3 · LangChain 与 RAG 入门"));
children.push(h3("学习内容"));
children.push(bullet("LangChain 核心抽象：Runnable / LCEL / Chain / Agent / Tool"));
children.push(bullet("Memory 类型：Buffer / Summary / VectorStore-backed"));
children.push(bullet("RAG 全流程：Loader → Splitter → Embedding → VectorStore → Retriever → Rerank"));
children.push(bullet("常用向量库对比：Chroma / Qdrant / Milvus / pgvector"));
children.push(h3("动手任务"));
children.push(bullet("用 LangChain 复刻 W2 的 Agent，对比代码量和可读性"));
children.push(bullet("搭建一个「研报问答」RAG：导入 10 份 PDF，能基于内容问答并标注来源"));

children.push(h2("W4 · Claude Agent SDK 与自建框架对比"));
children.push(h3("学习内容"));
children.push(bullet("Claude Agent SDK 核心理念：subagent、permission system、hooks"));
children.push(bullet("MCP（Model Context Protocol）协议解析：为什么它是「Tool 的标准化总线」"));
children.push(bullet("自建一个 100 行以内的极简 Agent 框架，作为理解所有大框架的基线"));
children.push(h3("动手任务"));
children.push(bullet("同一个「网页摘要 Agent」用 LangChain / Agent SDK / 原生三种方式实现"));
children.push(bullet("输出一份《三种实现的取舍对比》表格：抽象层级、可调试性、可观测性、生态、价格"));
children.push(bullet("基于对比结果选定本阶段后续主力栈（推荐主力：LangGraph + Claude Agent SDK 互补）"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// Phase 2
children.push(h1("四、第二阶段（W5-W8）：进阶能力 + 量化领域准备 + MVP"));

children.push(h2("W5 · LangGraph 与多 Agent 协作"));
children.push(h3("学习内容"));
children.push(bullet("状态机思维：把 Agent 看成「带共享 State 的有向图」"));
children.push(bullet("常见多 Agent 拓扑：Supervisor / Hierarchical / Network / Plan-and-Execute"));
children.push(bullet("Routing & Handoff：什么时候让另一个 Agent 接管，怎么传递上下文"));
children.push(h3("动手任务"));
children.push(bullet("用 LangGraph 实现 Planner-Worker 结构：Planner 拆解任务，Worker 执行，Reviewer 验收"));
children.push(bullet("在该 demo 上故意制造死循环、状态污染，学会用 checkpoint 和 interrupt 调试"));

children.push(h2("W6 · RAG 工程化与长期记忆"));
children.push(h3("学习内容"));
children.push(bullet("Chunking 策略：固定窗口 / 语义切分 / 父子文档 / 表格特化"));
children.push(bullet("Hybrid Search：BM25 + 向量 + Rerank（bge-reranker / cohere）"));
children.push(bullet("Memory 层级：Working Memory（当前对话）/ Episodic（事件）/ Semantic（事实）"));
children.push(bullet("Eval：RAGAS、TruLens、自建评估集"));
children.push(h3("动手任务"));
children.push(bullet("搭建「金融研报知识库 v1」：东方财富/巨潮资讯/雪球研报 ≥ 500 篇，支持引用回溯"));
children.push(bullet("构建 50 题评估集，用 RAGAS 跑出 baseline 分数"));

children.push(h2("W7 · 量化投资领域知识与基础设施"));
children.push(h3("领域知识速通"));
children.push(bullet("市场结构：A 股 / 港股 / 美股的差异、交易规则、数据频率"));
children.push(bullet("策略类型：趋势 / 均值回归 / 套利 / 因子 / 机器学习；Long-Short、市场中性"));
children.push(bullet("评价指标：年化收益、夏普、最大回撤、卡玛比、胜率、盈亏比、Calmar"));
children.push(bullet("常见坑：未来函数、幸存者偏差、过拟合、交易成本、滑点"));
children.push(h3("工具链选型"));
children.push(makeTable(
  [1800, 2800, 4760],
  ["类别", "推荐方案", "选用理由"],
  [
    ["数据源", "AKShare（免费）/ Tushare Pro / Wind", "AKShare 起步零成本，Tushare 适合做严肃研究"],
    ["回测框架", "Backtrader / vnpy / qlib", "Backtrader 上手快；qlib 来自微软，对 AI 策略友好"],
    ["数据存储", "DuckDB + Parquet（本地）/ PostgreSQL+TimescaleDB", "中小规模本地化优先，DuckDB 列存性能极佳"],
    ["分析栈", "pandas / polars / numpy / scipy / statsmodels", "polars 在大数据集下显著快于 pandas"],
    ["可视化", "Plotly / mplfinance / streamlit", "Streamlit 一键出策略报告页面"],
    ["LLM 编程", "Claude (代码) + DeepSeek (低成本) + 本地 7B (隐私)", "组合使用：贵的写代码，便宜的做批量任务"],
  ]
));
children.push(h3("动手任务"));
children.push(bullet("跑通 Backtrader 自带示例 + AKShare 拉一次 A 股数据 + 用 DuckDB 存储"));
children.push(bullet("写一个最朴素的双均线策略，跑出回测报告（不涉及 LLM，纯熟悉工具链）"));

children.push(h2("W8 · 量化 Agent MVP：自然语言到策略代码"));
children.push(h3("MVP 目标"));
children.push(p("用户用一句话「帮我写一个 20 日 60 日双均线策略，标的沪深 300，2019 年至今」，Agent 应当："));
children.push(bullet("解析意图 → 生成 Backtrader 策略代码"));
children.push(bullet("调用数据工具拉取数据"));
children.push(bullet("执行回测并产出图表与指标"));
children.push(bullet("用自然语言总结策略表现并指出潜在问题"));
children.push(h3("技术拆解"));
children.push(bullet("Tool 1：fetch_market_data(symbol, start, end, freq)"));
children.push(bullet("Tool 2：generate_strategy_code(spec) → 返回可执行 Python"));
children.push(bullet("Tool 3：run_backtest(code, data) → 返回指标 + 资金曲线图"));
children.push(bullet("Tool 4：critique_strategy(metrics, code) → LLM 自我反思 + 风险点"));
children.push(callout("MVP 验收标准",
  "能跑通 3 类基础策略：双均线 / RSI 反转 / 布林带突破，且任一策略全流程在 90 秒内出报告。MVP 不追求漂亮，追求闭环。"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// Phase 3
children.push(h1("五、第三阶段（W9-W12）：项目深化与产品化"));

children.push(h2("W9 · 因子挖掘 Agent"));
children.push(h3("能力目标"));
children.push(bullet("给定一个市场假设（如「小市值高换手」），自动生成候选因子表达式"));
children.push(bullet("调用 Code Interpreter 计算因子值、IC/IR、分层收益"));
children.push(bullet("自动剔除噪声因子，输出 Top-N 报告"));
children.push(h3("关键技术点"));
children.push(bullet("Symbolic Factor Mining：让 LLM 输出形式化的因子表达式（Alpha 101 风格）"));
children.push(bullet("沙箱执行：用 Docker 或 restrictedpython 跑 LLM 生成的代码"));
children.push(bullet("Self-Reflection：让 Agent 在跑完后批评自己的因子并迭代"));

children.push(h2("W10 · 回测 Agent + 评估循环"));
children.push(h3("能力目标"));
children.push(bullet("自动跑回测、自动绘图、自动生成 PDF/HTML 报告"));
children.push(bullet("基于结果反向优化：调参 / 加止损 / 换标的池"));
children.push(bullet("注入「未来函数检查」「过拟合检查」等专家规则"));
children.push(h3("关键技术点"));
children.push(bullet("Reflexion 范式：Actor + Evaluator + Self-Reflection 三层结构"));
children.push(bullet("评估集构建：人工标注 30 个「好策略 / 坏策略」样例作为 Agent 的对照基线"));
children.push(bullet("成本控制：哪些步骤用便宜模型（DeepSeek），哪些用 Claude/GPT-4 级别"));

children.push(h2("W11 · 多 Agent 编排：研究院级别工作流"));
children.push(p("把前面所有能力组装成一个仿真的量化研究院："));
children.push(makeTable(
  [2000, 7360],
  ["角色 Agent", "职责"],
  [
    ["首席研究员 (Supervisor)", "拆解用户需求，调度下游 Agent，汇总最终报告"],
    ["资讯分析师", "RAG 检索研报 / 新闻 / 公告，输出市场观点"],
    ["因子工程师", "基于观点生成因子假设并挖掘"],
    ["策略工程师", "把因子组合成策略代码"],
    ["回测工程师", "执行回测、生成报告、识别风险"],
    ["风控审查员", "用专家规则检查策略合规性与稳健性"],
  ]
));
children.push(p("技术实现：用 LangGraph 的 Supervisor 模式 + 共享 State + Checkpoint 持久化，每个角色用最合适的模型（不要无脑全 Claude）。"));

children.push(h2("W12 · 产品化、评估与复盘"));
children.push(h3("交付物清单"));
children.push(bullet("Streamlit / Next.js 前端，演示对话式量化研究"));
children.push(bullet("可观测性面板：每个 Agent 的调用次数、token 成本、平均耗时"));
children.push(bullet("自动评估脚本：跑预置 20 个任务，输出通过率与平均成本"));
children.push(bullet("一份 ≥ 5000 字的技术博客 / 项目复盘"));
children.push(bullet("一段 5 分钟产品演示视频"));
children.push(h3("评估标准"));
children.push(makeTable(
  [2400, 6960],
  ["维度", "及格线"],
  [
    ["闭环度", "20 个预置任务 ≥ 80% 跑通"],
    ["回答质量", "盲评中专业用户认为「可用」≥ 70%"],
    ["成本", "单次完整研究流程 ≤ $1（或等价）"],
    ["可观测", "出问题能在 5 分钟内定位到具体 Agent / 工具调用"],
    ["可扩展", "新增一个角色 Agent ≤ 1 天工作量"],
  ]
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// Project blueprint
children.push(h1("六、旗舰项目蓝图：量化投资 Agent"));

children.push(h2("6.1 产品定位"));
children.push(p("一个对话式的「私人量化研究助理」，面向个人投资者与初阶量化研究员。用户用自然语言描述想法，Agent 把想法翻译成可回测的策略，并给出有专业深度的反馈。它不是黑盒预测器，而是把研究员的工作流自动化的协作者。"));

children.push(h2("6.2 系统架构"));
children.push(p("分四层："));
children.push(richBullet([tn("交互层：", { bold: true }), tn("Streamlit / Web 前端，支持对话、文件上传、报告下载；")]));
children.push(richBullet([tn("编排层：", { bold: true }), tn("LangGraph Supervisor，负责任务分派、状态管理、Checkpoint；")]));
children.push(richBullet([tn("能力层：", { bold: true }), tn("6 个角色 Agent（见 W11），每个 Agent 内部还可以分解为子 Agent；")]));
children.push(richBullet([tn("基础设施层：", { bold: true }), tn("数据 API、向量库、沙箱执行器、监控/日志、模型路由。")]));

children.push(h2("6.3 数据与知识资产"));
children.push(bullet("行情数据：AKShare（A 股全量）+ yfinance（美股）→ DuckDB 本地化"));
children.push(bullet("基本面：财务报表、行业分类、ESG 标签"));
children.push(bullet("研报库：东方财富、巨潮资讯、雪球公开研报，定期增量入库"));
children.push(bullet("新闻舆情：财联社/华尔街见闻 RSS + 情感打分"));
children.push(bullet("策略知识：经典 Alpha 101、Worldquant Brain 公开因子、《主动投资组合管理》要点 chunk 化"));

children.push(h2("6.4 关键技术风险与对策"));
children.push(makeTable(
  [2600, 6760],
  ["风险", "对策"],
  [
    ["LLM 生成的代码不安全", "强制走 Docker 沙箱 + import 白名单 + 执行超时"],
    ["未来函数 / 数据穿越", "回测层强制使用 walk-forward + 自动检测可疑 shift(-N)"],
    ["过拟合", "样本外验证 + 参数敏感性分析作为标准产出"],
    ["Token 成本失控", "模型路由：编码用 Claude，批量摘要用 DeepSeek，本地任务用 7B"],
    ["可信度", "所有结论强制带数据来源与回测路径，可一键复现"],
  ]
));

children.push(h2("6.5 里程碑日历"));
children.push(makeTable(
  [1400, 7960],
  ["里程碑", "标志事件"],
  [
    ["M1（W4 末）", "三种框架对比完成，主力栈定型，第一版「能跑工具」的 Agent 就绪"],
    ["M2（W8 末）", "量化 Agent MVP 上线：自然语言 → 策略 → 回测 → 报告，闭环跑通"],
    ["M3（W10 末）", "因子挖掘 + 回测 Agent + 自我反思链路打通"],
    ["M4（W12 末）", "完整研究院多 Agent 系统 + 演示页面 + 公开复盘文章"],
  ]
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// Cross-domain extension
children.push(h1("七、能力迁移：从量化 Agent 到任意垂直领域"));
children.push(p("量化 Agent 的难点几乎覆盖了垂直 Agent 的所有典型问题：长文档理解、代码生成与执行、领域知识、严格评估、多角色协作、成本控制。一旦完成这一项目，向其他领域迁移基本是套模板的事。"));

children.push(h2("7.1 通用迁移方法论（5 步法）"));
children.push(bullet("Step 1：拆解领域工作流——找出该领域专家 1 天的真实任务清单"));
children.push(bullet("Step 2：识别可自动化的子任务（高重复、有明确输入输出）"));
children.push(bullet("Step 3：给每个子任务匹配 Agent 范式（RAG / Tool Use / Code Interp / Reflection）"));
children.push(bullet("Step 4：构建领域评估集（≥ 50 个真实任务+人类标注答案）"));
children.push(bullet("Step 5：从 MVP 单 Agent 起步，证明价值后再拓展为多 Agent 系统"));

children.push(h2("7.2 平移示例"));
children.push(makeTable(
  [2000, 3680, 3680],
  ["领域", "可直接复用的能力", "需要额外定制"],
  [
    ["法律合规", "RAG 研报 → RAG 法条/判例；研究员-审查员双 Agent", "条款级别对比工具、可解释引用、保密沙箱"],
    ["医疗科研", "因子挖掘 → 假设生成；回测 → 文献证据强度评估", "医学命名实体、PubMed 接入、人工复核闭环"],
    ["企业内研发", "策略代码生成 → 业务代码生成；多角色研究院 → 研发评审", "代码库 RAG、CI 集成、私有化部署"],
    ["客服与运营", "研究员 → 知识助理；分析师 → 工单分类与回执起草", "对话长度优化、情绪识别、低成本批量推理"],
    ["教育与培训", "评估循环 → 学生作答评测；自我反思 → 错题归因", "学情画像、节奏控制、防止 LLM 替考"],
  ]
));

children.push(h2("7.3 个人能力沉淀建议"));
children.push(bullet("每个新领域至少做一个 1 周 MVP，强迫自己提炼出「与上一个领域哪些不同 / 哪些相同」"));
children.push(bullet("维护一个个人 Agent 组件库（你的 prompt、tool、evaluator 模板），跨项目复用"));
children.push(bullet("定期写公开博客 / 在 GitHub 维护项目，把作品集变成简历"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// Resources
children.push(h1("八、推荐资源清单"));

children.push(h2("8.1 必读论文（按顺序）"));
children.push(bullet("ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)"));
children.push(bullet("Toolformer: Language Models Can Teach Themselves to Use Tools"));
children.push(bullet("Reflexion: Language Agents with Verbal Reinforcement Learning"));
children.push(bullet("Self-Refine: Iterative Refinement with Self-Feedback"));
children.push(bullet("Voyager: An Open-Ended Embodied Agent with LLMs"));
children.push(bullet("AutoGen / MetaGPT / CrewAI / LangGraph 的官方设计文档"));
children.push(bullet("Anthropic《Building Effective Agents》《How to build agentic systems》"));

children.push(h2("8.2 课程与系列文章"));
children.push(bullet("DeepLearning.AI × LangChain 系列短课"));
children.push(bullet("DeepLearning.AI × CrewAI / AutoGen 多 Agent 短课"));
children.push(bullet("Lilian Weng 博客《LLM Powered Autonomous Agents》"));
children.push(bullet("Eugene Yan 的 ML / LLM 系统化博客"));
children.push(bullet("Anthropic Engineering 博客（持续更新）"));

children.push(h2("8.3 开源项目精读"));
children.push(bullet("LangGraph：现代 Agent 编排参考实现"));
children.push(bullet("Claude Agent SDK 开源 reference 代码"));
children.push(bullet("MetaGPT：多角色 Agent 范式"));
children.push(bullet("OpenDevin / SWE-Agent：代码 Agent 工程范式"));
children.push(bullet("Qlib：量化投资框架，AI-friendly 数据 schema"));
children.push(bullet("FinGPT / FinRobot：金融领域 Agent 参考"));

children.push(h2("8.4 量化领域专属"));
children.push(bullet("《主动投资组合管理》Grinold & Kahn（信号、IC、信息比率经典）"));
children.push(bullet("《量化投资策略》Inmon（中文译本，工程视角）"));
children.push(bullet("Worldquant Brain 公开因子库"));
children.push(bullet("聚宽 / Ricequant / 优矿 教程区"));
children.push(bullet("AQR Capital 公开研究论文"));

children.push(h2("8.5 工具与平台"));
children.push(bullet("模型：Claude（推理/代码强）、GPT-4o、DeepSeek（性价比）、Qwen/Llama（本地）"));
children.push(bullet("向量库：Chroma（轻量）/ Qdrant（生产）/ pgvector（与 RDB 共存）"));
children.push(bullet("观测：LangSmith / Langfuse / Phoenix（Arize）"));
children.push(bullet("评估：RAGAS / TruLens / Promptfoo / 自建评估集"));
children.push(bullet("沙箱：E2B / Modal / Docker；私有部署可用 firejail"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// Weekly cadence & habits
children.push(h1("九、每周节奏与执行守则"));

children.push(h2("9.1 每周时间分配（约 30 小时）"));
children.push(makeTable(
  [2400, 1600, 5360],
  ["类别", "时间", "做什么"],
  [
    ["原理学习", "6h", "看论文 / 课程 / 博客，做笔记，能复述给别人听"],
    ["动手实践", "16h", "本周项目代码，跑通 demo，记录踩坑"],
    ["读源码", "3h", "精读一个开源 Agent 的关键模块"],
    ["复盘输出", "3h", "周末写本周小结（笔记/博客），形成可分享的内容"],
    ["弹性时间", "2h", "处理意外、修旧坑、社区交流"],
  ]
));

children.push(h2("9.2 学习守则"));
children.push(bullet("不允许「只看不写」：每周末必须有 push 到 GitHub 的代码"));
children.push(bullet("不允许「只写不评」：每个 demo 必须有一段失败案例与反思"));
children.push(bullet("不允许「黑盒框架」：用任何框架前先讲清它的核心抽象与替代方案"));
children.push(bullet("每两周做一次「假装在面试」自测：能否 15 分钟讲清本周项目的设计？"));

children.push(h2("9.3 风险预警与节奏调整"));
children.push(callout("如果某一周严重落后",
  "不要顺延所有内容。优先级永远是：保住本周「能跑的产物」 > 完成所有阅读。把读不完的论文挪到第 13 周补，但 demo 不能跳。"));

children.push(h2("9.4 评估自己是否「资深」"));
children.push(p("到 W12 结束时，用下列问题自检（能答 ≥ 8 个即达到「准资深」水平）："));
children.push(bullet("能用一张图讲清 ReAct / Reflexion / Plan-and-Execute 的差异？"));
children.push(bullet("给定一个新需求，能在 30 分钟内画出 Agent 系统架构？"));
children.push(bullet("LangChain 和 Claude Agent SDK 各自的优势场景说得清？"));
children.push(bullet("RAG 出现幻觉时，你的 5 个排查方向是什么？"));
children.push(bullet("如何防止 LLM 生成代码逃逸沙箱？说出三种机制"));
children.push(bullet("Token 成本下降 50% 的方案能给出 ≥ 3 个？"));
children.push(bullet("如何为一个新领域 Agent 构建评估集？"));
children.push(bullet("多 Agent 死锁/循环的常见诱因和拆解方法？"));
children.push(bullet("能讲清 MCP 协议的设计动机？"));
children.push(bullet("能现场写 100 行实现一个最小 Agent？"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// Closing
children.push(h1("十、写在最后"));
children.push(p("这条 12 周路线不是「教程清单」，而是一条「逼自己产出」的训练带。它的价值在于：当你 12 周后回头看，你能拿出一个真实可演示的产品、一套自己的方法论、一个开源仓库、若干篇技术博客——这些才是别人判断你是不是「资深」的依据，而不是看过多少课。"));
children.push(p("量化投资 Agent 只是入口。当你完成这一个，去做法律、医疗、研发任何一个垂直领域的 Agent，本质上只是换了一组工具和评估集——核心的 Agent 设计能力、工程能力、评估能力都是同一套。"));
children.push(p("祝顺利。——Claude"));

// ---------- Build document ----------
const doc = new Document({
  creator: "Claude",
  title: "Agent 学习与训练路线",
  styles: {
    default: { document: { run: { font: FONT_CN, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: FONT_CN, color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT_CN, color: "2E75B6" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT_CN, color: "385723" },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
        ],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "Agent 学习与训练路线 · 量化投资实战", font: FONT_CN, size: 18, color: "808080", italics: true })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "第 ", font: FONT_CN, size: 18, color: "808080" }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT_CN, size: 18, color: "808080" }),
            new TextRun({ text: " 页 · 共 ", font: FONT_CN, size: 18, color: "808080" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT_CN, size: 18, color: "808080" }),
            new TextRun({ text: " 页", font: FONT_CN, size: 18, color: "808080" }),
          ],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = "/sessions/loving-elegant-archimedes/mnt/outputs/Agent学习与训练路线_量化投资实战.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Wrote:", outPath, "size:", buffer.length);
});
