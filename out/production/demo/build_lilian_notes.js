// Lilian Weng "LLM Powered Autonomous Agents" — distilled study notes
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  ExternalHyperlink,
} = require('docx');

const FONT = "Microsoft YaHei";
const border = { style: BorderStyle.SINGLE, size: 6, color: "B5C7D8" };
const allBorders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 100, bottom: 100, left: 140, right: 140 };

function tn(text, opts = {}) {
  return new TextRun({
    text, font: FONT, size: opts.size || 22,
    bold: opts.bold, color: opts.color, italics: opts.italics,
  });
}
function code(text) {
  return new TextRun({ text, font: "Consolas", size: 20, color: "BF360C" });
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
  const bodyParas = (Array.isArray(body) ? body : [body]).map(t =>
    new Paragraph({ children: [tn(t, { color: "3F3000" })], spacing: { line: 300, after: 40 } })
  );
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
        new Paragraph({ children: [tn(title, { bold: true, color: "5F4B00" })], spacing: { after: 60 } }),
        ...bodyParas,
      ],
    })]})],
  });
}
function makeTable(columnWidths, headerRow, rows, opts = {}) {
  const tableWidth = columnWidths.reduce((a, b) => a + b, 0);
  const headerCells = headerRow.map((text, i) =>
    new TableCell({
      width: { size: columnWidths[i], type: WidthType.DXA },
      shading: { fill: opts.headerFill || "1F4E79", type: ShadingType.CLEAR },
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

const children = [];

// Cover
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 600, after: 100 },
  children: [tn("LLM Powered Autonomous Agents", { size: 48, bold: true, color: "1F4E79" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [tn("Lilian Weng · 2023.06 · 速通版学习卡片", { size: 26, color: "2E75B6" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 360 },
  children: [
    tn("原文：", { size: 20, color: "808080" }),
    new ExternalHyperlink({
      children: [new TextRun({ text: "lilianweng.github.io/posts/2023-06-23-agent", font: FONT, size: 20, color: "0563C1", underline: { type: "single" } })],
      link: "https://lilianweng.github.io/posts/2023-06-23-agent/",
    }),
  ],
}));

// One-sentence summary
children.push(callout(
  "一句话总览",
  "Agent = LLM（大脑） + Planning（规划） + Memory（记忆） + Tool Use（工具）。LLM 在这套架构里不只是「文本生成器」，而是「通用问题求解器」的中枢。",
  "EAF3FA", "2E75B6"
));

children.push(callout(
  "核心心智模型（请记住这张图）",
  [
    "把 Agent 看作一个 while 循环：用户目标 → LLM 思考（含读 Memory）→ 选择 Tool 调用 → 观察结果 → 更新 Memory → 继续思考 → 完成或终止。",
    "三大支柱（Planning / Memory / Tool）正是这个循环的三个抽象功能槽。",
  ],
  "FFF4CE", "F0C040"
));

children.push(h2("阅读完本卡片你应当能回答"));
children.push(bullet("Agent 的三大组件分别解决什么问题？"));
children.push(bullet("CoT、ToT、ReAct、Reflexion 各自的演化关系？"));
children.push(bullet("为什么 Memory 必须分短期与长期？两者技术实现差异？"));
children.push(bullet("MIPS 算法 5 强（LSH / ANNOY / HNSW / FAISS / ScaNN）的取舍？"));
children.push(bullet("Tool Use 在 LLM 视角下的难点是什么？API-Bank 三级评估测什么？"));
children.push(bullet("当前 LLM Agent 的三大根本性局限？"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// =============== 第一部分：总览 ===============
children.push(h1("一、Agent 系统总览"));
children.push(p("LLM 充当 Agent 的「大脑」，外围补三个能力，从而把「会说话」升级为「会做事」。"));
children.push(makeTable(
  [1400, 2400, 5560],
  ["组件", "子能力", "本质 / 解决的问题"],
  [
    ["Planning 规划", "子目标拆解", "把模糊大任务变成可执行的小步骤，约束 LLM 一次只想一步"],
    ["", "反思与改进", "用「失败 → 反思 → 重试」对抗 LLM 一次推理失败的不稳定性"],
    ["Memory 记忆", "短期记忆", "对应 In-Context Learning：上下文窗口内的临时信息"],
    ["", "长期记忆", "外部向量库 + 快速检索，突破上下文长度限制"],
    ["Tool Use 工具", "调用外部 API", "弥补 LLM 不会的事：算数、查时效信息、跑代码、调专有系统"],
  ]
));
children.push(callout(
  "关键洞察",
  "三大组件不是「加法」而是「乘法」——缺任何一个，Agent 都退化成不同程度的「话痨」：缺规划=只会一步，缺记忆=失忆症，缺工具=只能空想。",
  "EAF3FA", "2E75B6"
));

// =============== 第二部分：Planning ===============
children.push(h1("二、Planning（规划）"));

children.push(h2("2.1 任务分解（Task Decomposition）"));
children.push(p("解决「大任务一口气想不清」。四种主流路径："));

children.push(makeTable(
  [1600, 2400, 5360],
  ["方法", "核心思想", "适用场景与局限"],
  [
    ["CoT（Wei 2022）", "Prompt 让模型「step by step」，把长推理拆成中间步", "通用提升；但仍是单线推理，错误会传播"],
    ["ToT（Yao 2023）", "每步生成多个候选 thought，用 BFS/DFS + 评分搜索", "复杂规划/对弈类任务；token 成本高"],
    ["LLM+P（Liu 2023）", "LLM 把问题翻成 PDDL → 外部经典 planner 求解 → 翻回自然语言", "机器人等已有 domain PDDL 的领域，其他领域难落地"],
    ["人工/任务专属指令", "直接给指令模板，如「写小说先列大纲」", "可控性最强，但需要为每类任务设计"],
  ]
));
children.push(callout(
  "演化逻辑",
  "CoT（单线）→ ToT（多线 + 搜索）→ LLM+P（外挂经典 planner）。复杂度逐级上升，每一级解决前一级的失败点。",
  "EAF3FA", "2E75B6"
));

children.push(h2("2.2 自我反思（Self-Reflection）"));
children.push(p("解决「LLM 一旦走错就一路错下去」。四个代表方法（演化关系比方法本身更重要）："));

children.push(makeTable(
  [1400, 7960],
  ["方法", "核心机制"],
  [
    ["ReAct（Yao 2022）", "把动作空间扩为「自然语言推理 + 离散动作」，prompt 形如：Thought → Action → Observation → 循环。所有 Agent 的祖师爷。"],
    ["Reflexion（Shinn 2023）", "ReAct 之上加 RL 风格 + 启发式：每次失败后生成「文字反思」，写进 working memory；启发式检测低效轨迹（太长）与幻觉（重复同动作得同观察）"],
    ["CoH（Liu 2023）", "把「过去多个版本的输出 + 评分 + 修改建议」打包进 prompt，让模型「看着自己历次失败学着变好」（fine-tune 而非纯 prompt）"],
    ["AD（Laskin 2023）", "Algorithm Distillation：把多次 RL 训练历史拼成长序列让模型学习「学习算法本身」；in-context RL"],
  ]
));

children.push(h3("ReAct 模板（必背）"));
children.push(p([
  code("Thought:"), tn(" 我应该…"),
]));
children.push(p([
  code("Action:"), tn(" search[\"...\"]"),
]));
children.push(p([
  code("Observation:"), tn(" ...（外部环境返回）"),
]));
children.push(p("…（重复 N 次直到答案出现）"));

children.push(h3("Reflexion 启发式的两条「停手」规则"));
children.push(bullet("低效规划：轨迹太长仍未成功 → 触发反思 + 重启"));
children.push(bullet("幻觉：连续相同动作得到相同观察 → 模型陷入循环，触发反思"));

children.push(callout(
  "实战映射",
  "在你的量化 Agent 里：策略代码生成失败、回测指标异常、Agent 反复尝试同一种修复——都该用 Reflexion 思路触发自我反思而非硬继续。",
  "EAF3FA", "2E75B6"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// =============== 第三部分：Memory ===============
children.push(h1("三、Memory（记忆）"));

children.push(h2("3.1 三种记忆类型 → LLM 工程的映射"));
children.push(p("Lilian 把人脑记忆模型类比到 LLM 系统，这是全文最有价值的「映射表」之一："));

children.push(makeTable(
  [1800, 3200, 4360],
  ["人脑记忆类型", "特征", "在 LLM 系统中的对应物"],
  [
    ["Sensory Memory（感官）", "原始感知瞬时痕迹（<1 秒）", "Embedding：对原始 token/图像/语音的向量表示"],
    ["Short-Term / Working", "约 7 项，持续 20-30 秒", "In-Context Learning：上下文窗口内的内容"],
    ["Long-Term（显式）", "事实与事件，可几十年", "外部向量库 + 检索（RAG / Memory Store）"],
    ["Long-Term（隐式 / 程序）", "技能、习惯，无意识", "模型权重本身（pre-training 学到的能力）"],
  ]
));

children.push(callout(
  "关键观点",
  "上下文窗口 ≠ 长期记忆。无论多大的窗口（200K / 1M），都是「短期工作记忆」。要让 Agent 真正记得久，必须显式上向量库。",
  "EAF3FA", "2E75B6"
));

children.push(h2("3.2 MIPS 算法 5 强（向量检索核心）"));
children.push(p("长期记忆要工程化，关键是「快速最大内积搜索（MIPS）」。下面 5 个是工业界主流："));

children.push(makeTable(
  [1100, 3600, 4660],
  ["算法", "核心思想", "取舍 / 何时选"],
  [
    ["LSH", "哈希函数让相似项落同桶，桶数 << 数据量", "实现简单；高维高召回时性能下降"],
    ["ANNOY", "随机投影树森林，搜索时聚合多棵树", "Spotify 出品，工程稳定；构建较慢"],
    ["HNSW", "分层小世界图，上层跳长距、下层精搜", "目前生产首选，召回率与速度俱佳"],
    ["FAISS", "向量量化分簇 + 簇内细量化（IVF + PQ）", "Facebook 出品，海量数据强项；调参较多"],
    ["ScaNN", "各向异性量化：保留内积而非欧氏距离", "Google 出品，召回率领先；适合 inner-product 任务"],
  ]
));

children.push(callout(
  "选型建议",
  "MVP 阶段用 Chroma/Qdrant（底层多用 HNSW），不用纠结。等数据 ≥ 1000 万条或延迟敏感时再考虑 FAISS/ScaNN 调优。",
  "FFF4CE", "F0C040"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// =============== 第四部分：Tool Use ===============
children.push(h1("四、Tool Use（工具使用）"));

children.push(h2("4.1 演化谱系"));
children.push(makeTable(
  [1800, 7560],
  ["代表方法", "贡献点"],
  [
    ["MRKL（Karpas 2022）", "「LLM 当路由 + 专家模块（神经/符号）」的范式起点；指出关键不是「会调」而是「知道何时调」"],
    ["TALM / Toolformer（2022-23）", "Fine-tune LLM 学习何时插入 API 调用；自训练数据筛选机制"],
    ["ChatGPT Plugins / Function Calling", "把上述能力工程化、协议化（OpenAI 2023）"],
    ["HuggingGPT（Shen 2023）", "LLM 当任务规划器，调度 HuggingFace 上的专家模型；四阶段范式"],
    ["API-Bank（Li 2023）", "评估基准：53 个 API、264 个对话、568 次调用；三级评估体系"],
  ]
));

children.push(h2("4.2 HuggingGPT 四阶段范式（值得背下来）"));
children.push(bullet("① Task planning：LLM 把用户输入拆为带依赖关系的子任务 JSON 列表"));
children.push(bullet("② Model selection：每个子任务用多选题方式让 LLM 选最适合的专家模型"));
children.push(bullet("③ Task execution：调用专家模型，记录结果"));
children.push(bullet("④ Response generation：汇总执行结果生成最终回答"));
children.push(callout(
  "工程难点 3 条",
  "（1）效率：多轮 LLM + 多次模型调用慢；（2）依赖长上下文；（3）LLM 与外部服务输出的稳定性问题——绝大多数 demo 代码都在做 output parsing。",
  "EAF3FA", "2E75B6"
));

children.push(h2("4.3 API-Bank 三级评估（设计你自己评估集的模板）"));
children.push(bullet("L1 调用：会不会调？参数对不对？能不能用返回结果回答？"));
children.push(bullet("L2 检索：会不会从大量 API 里找出该用哪个？读不读得懂文档？"));
children.push(bullet("L3 规划：需求模糊时，能不能多步调用 + 串联结果（如：订机票+酒店+餐厅）"));

children.push(callout(
  "实战映射",
  "你的量化 Agent 评估集照搬这三级：L1—会不会调 fetch_market_data？L2—10 个相似的数据 API 选对没？L3—「分析消费板块过去三年表现」能不能拆成数据+计算+回测三步？",
  "FFF4CE", "F0C040"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// =============== 第五部分：案例 ===============
children.push(h1("五、四个标杆案例"));

children.push(h2("5.1 ChemCrow（化学领域 Agent）"));
children.push(bullet("13 个专家工具 + ReAct 格式 + LangChain 实现"));
children.push(bullet("Tool 描述要包含名称、用途、I/O；这套模板适用于所有垂直领域"));
children.push(callout(
  "最重要的发现",
  "LLM 自评 GPT-4 ≈ ChemCrow，但化学家专家评估 ChemCrow 远胜 GPT-4。结论：用 LLM 评估自己在专家领域的输出是不可靠的——必须引入领域专家或专家规则。",
  "EAF3FA", "2E75B6"
));

children.push(h2("5.2 Boiko et al.（科研 Agent）"));
children.push(bullet("能浏览网页、读文档、执行代码、调机器人、调用其他 LLM"));
children.push(bullet("「设计抗癌药」演示：查趋势 → 选靶点 → 设计骨架 → 尝试合成"));
children.push(callout(
  "安全警示",
  "11 个化武合成请求中，36% 通过未被拒绝——LLM Agent 的滥用风险是必须考虑的设计维度。给你的量化 Agent 设置安全边界（不许造谣、不许操纵市场建议、不许越权访问账户）。",
  "FFE6E6", "C00000"
));

children.push(h2("5.3 Generative Agents（25 智能体小镇）"));
children.push(p("一个 Sims 风格的沙箱，每个虚拟角色是一个 LLM Agent。架构里有四个值得记的概念："));
children.push(makeTable(
  [1800, 7560],
  ["机制", "做什么"],
  [
    ["Memory Stream", "自然语言记录所有观察事件，作为外部长期记忆"],
    ["Retrieval", "三因素打分：Recency 时近 + Importance 重要 + Relevance 相关；综合后取 top-k"],
    ["Reflection", "周期性把最近 100 条观察 → 让 LLM 提 3 个高级问题 → 自答 → 形成高级摘要写回记忆"],
    ["Planning & Reacting", "把反思 + 环境信息 → 生成行动；环境用 tree 结构表示"],
  ]
));
children.push(callout(
  "可借鉴点",
  "Retrieval 的 Recency × Importance × Relevance 三因素打分，是工业级 Memory 检索的最常用 baseline。",
  "EAF3FA", "2E75B6"
));

children.push(h2("5.4 AutoGPT / GPT-Engineer（早期 PoC）"));
children.push(bullet("AutoGPT：通过 system prompt 给 20 个命令 + 反思机制，全自动循环；问题是「极不稳定」，大量代码都在做 output parsing"));
children.push(bullet("GPT-Engineer：先用一个 system prompt 进行澄清问答，再换 system prompt 进入代码生成模式——「分阶段切换 prompt」是个朴素但有效的模式"));
children.push(callout(
  "教训",
  "完全放手的「全自动」目前不可靠。生产可用的范式更接近「人在回路 + 阶段化 prompt + 强约束 schema」。",
  "FFF4CE", "F0C040"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// =============== 第六部分：三大挑战 ===============
children.push(h1("六、当前 LLM Agent 的三大根本性挑战"));

children.push(makeTable(
  [2000, 7360],
  ["挑战", "本质"],
  [
    ["有限上下文长度", "历史信息、指令、API 上下文、响应都被挤压。即便有 RAG，向量检索的表达力仍不及 full attention"],
    ["长时规划与任务分解难", "LLM 难以面对意外错误时动态调整计划；缺乏人类「试错-学习」的稳健性"],
    ["自然语言接口不可靠", "格式错误、不听话、JSON 解析失败……大量 demo 代码都在为这个问题打补丁"],
  ]
));

children.push(callout(
  "对你的设计意味着什么",
  [
    "1) 不要指望「长上下文」解决一切——必须显式工程化 Memory；",
    "2) 不要做「一步到位」的复杂规划，更可靠的是「短跳 + 反思 + 重试」；",
    "3) 永远假设 LLM 输出会出错——用 Pydantic/JSON Schema 做强约束 + 重试 + 降级路径。",
  ],
  "EAF3FA", "2E75B6"
));

// =============== 第七部分：实战映射 ===============
children.push(h1("七、与「量化投资 Agent」的实战映射"));
children.push(p("把这篇文章每一节的方法对号入座到你的项目里："));

children.push(makeTable(
  [2400, 6960],
  ["文章概念", "你的量化 Agent 中对应位置"],
  [
    ["子目标拆解（CoT/ToT）", "Supervisor Agent 把「研究消费板块」拆为：取数 → 因子 → 策略 → 回测 → 风控"],
    ["LLM+P", "策略代码生成时，把策略规范翻成 Strategy DSL（你的 PDDL），由 Backtrader 执行"],
    ["ReAct", "所有子 Agent 内部循环：思考 → 调用 Tool → 观察 → 继续"],
    ["Reflexion", "回测 Agent 在策略失败时自我反思（指标差 / 报错 / 未来函数嫌疑）"],
    ["短期记忆", "当前对话状态、当前研究任务的中间结果"],
    ["长期记忆", "研报库 + 历次实验日志 + 用户偏好（向量库 + Recency/Importance/Relevance 三因素打分）"],
    ["MIPS 算法", "MVP 用 Chroma（HNSW）；500 篇研报 → 5 万篇时考虑 Qdrant/FAISS"],
    ["MRKL 范式", "Supervisor 是路由器，6 个角色 Agent 是专家模块"],
    ["HuggingGPT 四阶段", "你的端到端流程几乎一一对应：解析需求 → 选 Agent → 执行 → 汇总报告"],
    ["API-Bank 三级评估", "你的评估集照搬：L1 单工具、L2 多工具选择、L3 多步规划"],
    ["ChemCrow 教训", "回测结果绝不让 LLM 自评——必须用专家规则集 + 量化指标硬卡"],
    ["Generative Agents 反思", "周期性让风控审查员把多条历次实验 → 提炼为高级风控规则"],
  ]
));

children.push(callout(
  "如果你只能记 5 件事",
  [
    "1) Agent = LLM + Planning + Memory + Tool；",
    "2) ReAct 是循环骨架，Reflexion 是循环里的「自我纠错」补丁；",
    "3) Memory 必须分层（短期=context，长期=向量库），三因素打分检索；",
    "4) Tool 难的是「知道何时调」而非「能调用」；评估必须分 L1/L2/L3 三级；",
    "5) 别让 LLM 自评专业领域的输出——用专家规则或专家 in the loop。",
  ],
  "FFF4CE", "F0C040"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// =============== 第八部分：必读论文清单 ===============
children.push(h1("八、文章引用的必读论文（按读取优先级）"));

children.push(makeTable(
  [800, 4400, 4160],
  ["级别", "论文 / 来源", "读法建议"],
  [
    ["必读", "ReAct (Yao 2022, arxiv 2210.03629)", "三遍法精读，复述算法 + 写一遍 100 行实现"],
    ["必读", "Reflexion (Shinn 2023, 2303.11366)", "三遍法精读，重点理解启发式与 working memory"],
    ["必读", "CoT (Wei 2022, 2201.11903)", "第二遍即可，理解为什么 step-by-step 有效"],
    ["重要", "ToT (Yao 2023, 2305.10601)", "第二遍 + 跑一个 ToT 示例"],
    ["重要", "Toolformer (Schick 2023, 2302.04761)", "了解 self-training Tool Use 的思路"],
    ["重要", "HuggingGPT (Shen 2023, 2303.17580)", "重点记住四阶段范式与三个工程难点"],
    ["重要", "API-Bank (Li 2023, 2304.08244)", "重点记住三级评估，作为你评估集的模板"],
    ["重要", "Generative Agents (Park 2023, 2304.03442)", "重点看 Retrieval 三因素打分 + Reflection 机制"],
    ["选读", "LLM+P (Liu 2023, 2304.11477)", "第二遍即可，了解 PDDL 思路"],
    ["选读", "CoH (Liu 2023, 2302.02676)", "第二遍即可"],
    ["选读", "Algorithm Distillation (Laskin 2023, 2210.14215)", "纯研究兴趣可读"],
    ["选读", "MRKL (Karpas 2022, 2205.00445)", "短文，了解神经-符号架构起源"],
    ["选读", "ChemCrow (Bran 2023, 2304.05376)", "看垂直领域 Agent 工程范本"],
  ]
));

children.push(callout(
  "学习建议",
  "这篇博客是「目录式综述」。最高效的精读方式：读完本卡片 → 直接读 ReAct + Reflexion 两篇原文 → 立刻动手写 W2 的 100 行 ReAct → 再回头补 Reflexion 论文实现到 W10 的回测反思 Agent。",
  "EAF3FA", "2E75B6"
));

// =============== 附录：知识库卡片建议 ===============
children.push(h1("九、写进你的个人知识库（建议卡片）"));
children.push(p("按主文档的 KB 目录结构，本文应当沉淀以下卡片："));

children.push(h3("10_Concepts（原子概念）"));
children.push(bullet("Planning / Subgoal Decomposition / Self-Reflection"));
children.push(bullet("Short-Term vs Long-Term Memory（含 LLM 系统映射）"));
children.push(bullet("MIPS / ANN / HNSW / FAISS / ScaNN"));
children.push(bullet("Tool Use / MRKL / Function Calling"));

children.push(h3("20_Patterns（设计模式）"));
children.push(bullet("ReAct Loop（含模板）"));
children.push(bullet("Reflexion Loop（含两条启发式）"));
children.push(bullet("HuggingGPT 四阶段范式"));
children.push(bullet("Retrieval 三因素打分（Recency / Importance / Relevance）"));
children.push(bullet("API-Bank 三级评估"));

children.push(h3("40_Papers（论文卡）"));
children.push(bullet("一篇综述卡：Lilian Weng《LLM Powered Autonomous Agents》"));
children.push(bullet("ReAct / Reflexion / CoT / HuggingGPT / API-Bank 各一张论文卡"));

children.push(h3("70_Projects（项目映射）"));
children.push(bullet("把本卡片第七章「实战映射」表格直接复制进 70_Projects/quant_agent/design_decisions.md"));

// Build
const doc = new Document({
  creator: "Claude",
  title: "Lilian Weng LLM Agent 速通卡片",
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
        children: [tn("Lilian Weng《LLM Powered Autonomous Agents》· 速通卡片", { size: 18, color: "808080", italics: true })],
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
  const out = "/sessions/loving-elegant-archimedes/mnt/outputs/Lilian_Weng_Agent速通卡片.docx";
  fs.writeFileSync(out, buffer);
  console.log("Wrote:", out, "size:", buffer.length);
});
