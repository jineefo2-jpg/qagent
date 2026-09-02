// Anthropic Prompt Engineering Overview + Best Practices — distilled study notes
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  ExternalHyperlink,
} = require('docx');

const FONT = "Microsoft YaHei";
const MONO = "Consolas";
const border = { style: BorderStyle.SINGLE, size: 6, color: "B5C7D8" };
const allBorders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 100, bottom: 100, left: 140, right: 140 };

function tn(text, opts = {}) {
  return new TextRun({
    text, font: opts.mono ? MONO : FONT, size: opts.size || 22,
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
function codeBlock(text) {
  // a single-cell light-gray block, monospace
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: 9360, type: WidthType.DXA },
      shading: { fill: "F4F4F4", type: ShadingType.CLEAR },
      margins: { top: 120, bottom: 120, left: 180, right: 180 },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 4, color: "D0D0D0" },
        bottom: { style: BorderStyle.SINGLE, size: 4, color: "D0D0D0" },
        left: { style: BorderStyle.SINGLE, size: 4, color: "D0D0D0" },
        right: { style: BorderStyle.SINGLE, size: 4, color: "D0D0D0" },
      },
      children: text.split("\n").map(line =>
        new Paragraph({ children: [tn(line || " ", { mono: true, size: 20, color: "262626" })], spacing: { line: 260 } })
      ),
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

const children = [];

// ===== Cover =====
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 600, after: 100 },
  children: [tn("Prompt Engineering 速通卡片", { size: 48, bold: true, color: "1F4E79" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [tn("Anthropic 官方 Overview + Best Practices · 精炼版", { size: 26, color: "2E75B6" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 360 },
  children: [
    tn("原文 1：", { size: 20, color: "808080" }),
    new ExternalHyperlink({
      children: [new TextRun({ text: "docs.anthropic.com/.../prompt-engineering/overview", font: FONT, size: 20, color: "0563C1", underline: { type: "single" } })],
      link: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview",
    }),
  ],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 360 },
  children: [
    tn("原文 2：", { size: 20, color: "808080" }),
    new ExternalHyperlink({
      children: [new TextRun({ text: "docs.anthropic.com/.../claude-prompting-best-practices", font: FONT, size: 20, color: "0563C1", underline: { type: "single" } })],
      link: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices",
    }),
  ],
}));

// ===== TOC ish summary =====
children.push(callout(
  "一句话总览",
  "Prompt 工程 = 三件事：① 把任务讲清（角色、目标、格式）；② 把上下文喂对（XML 标签、示例、长文档放前面）；③ 把 LLM 的「自主程度」调到合适（thinking、effort、tool 触发、subagent）。",
  "EAF3FA", "2E75B6"
));

children.push(callout(
  "金科玉律（Golden Rule）",
  "把你的 Prompt 交给一个对任务没背景的同事来执行——如果他会困惑，Claude 也会。Claude 是「聪明但刚入职的新员工」，越具体越好。",
  "FFF4CE", "F0C040"
));

children.push(h2("阅读完本卡片你应当能"));
children.push(bullet("写出一个结构良好的 Prompt（角色 / 上下文 / 示例 / 输入 / 输出格式 / 检查清单 都有 XML 标签）"));
children.push(bullet("讲清 few-shot、role、XML、long-context、prefill 这些技巧分别解决什么问题"));
children.push(bullet("知道何时该用 thinking / adaptive thinking / effort 参数，以及如何控制 verbosity"));
children.push(bullet("能为 Agent 系统写一个能稳定触发并行工具调用、不过度生成 subagent 的 system prompt"));
children.push(bullet("识别新一代模型（4.6+）相比旧模型的 6 个行为变化，并知道如何针对性调整 Prompt"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 1: Overview =====
children.push(h1("一、Overview：什么时候才该「调 Prompt」"));
children.push(p("Overview 页本身极短，传递了两个判断："));
children.push(richBullet([tn("先决条件：", { bold: true }), tn("调 Prompt 之前必须有 ①明确的成功标准 ②能跑的评估 ③一份初稿 Prompt。没有这三样别开始调。")]));
children.push(richBullet([tn("Prompt 不是万能：", { bold: true }), tn("延迟、成本问题——换更小的模型可能比调 Prompt 更划算；事实性 RAG 问题——加检索比调 Prompt 更划算。Prompt 工程只解决「可被指令控制」的问题。")]));

children.push(callout(
  "工作流建议",
  "Console 里有 3 个官方工具值得用：Prompt Generator（生成初稿）、Templates & Variables（参数化复用）、Prompt Improver（自动改写）。Tutorial 在 GitHub anthropics/prompt-eng-interactive-tutorial，是最快入门路径。",
  "EAF3FA", "2E75B6"
));

// ===== Part 2: 通用原则 =====
children.push(h1("二、通用原则（6 大支柱）"));

children.push(h2("2.1 Be Clear and Direct（清晰直接）"));
children.push(bullet("把 Claude 当作「聪明但缺背景的新员工」"));
children.push(bullet("指定输出格式与约束；顺序敏感的步骤用编号列表"));
children.push(bullet("想要「超额完成」就明说，不要寄希望于模型自己推断"));

children.push(h2("2.2 Add Context（解释「为什么」）"));
children.push(p("把指令背后的动机告诉模型，模型能从中泛化。比起冷冰冰的「不要用 markdown」，写「为了能直接复制到邮件里，请用纯文本段落」效果更好。"));

children.push(h2("2.3 Use Examples（多示例 / Few-shot）"));
children.push(p("示例是最稳的格式控制器。三条原则："));
children.push(richBullet([tn("Relevant 相关：", { bold: true }), tn("贴近真实用例")]));
children.push(richBullet([tn("Diverse 多样：", { bold: true }), tn("覆盖边界情况，避免模型抓到错误模式")]));
children.push(richBullet([tn("Structured 结构化：", { bold: true }), tn("用 "), tn("<example>", { mono: true }), tn(" 包裹（多条用 "), tn("<examples>", { mono: true }), tn("），与指令明显区分")]));
children.push(bullet("推荐 3-5 个示例；可以让 Claude 自己评估示例多样性，甚至生成补充示例"));

children.push(h2("2.4 Structure with XML Tags（XML 结构化）"));
children.push(p("当 Prompt 里同时存在「指令 + 上下文 + 示例 + 变量输入」时，XML 标签是消歧利器。最常见的标签："));
children.push(bullet("<instructions> 指令"));
children.push(bullet("<context> 背景"));
children.push(bullet("<examples> / <example> 示例"));
children.push(bullet("<input> 用户输入"));
children.push(bullet("<documents> / <document index=\"n\"> 多文档"));
children.push(bullet("<thinking> / <answer> 思考与答案分离"));
children.push(callout(
  "实战诀窍",
  "在 Prompt 里用什么 XML 标签，Claude 输出时就倾向于用什么；如果想让输出是平实散文，Prompt 也别堆满 markdown 和 bullets——「Prompt 风格匹配输出风格」。",
  "EAF3FA", "2E75B6"
));

children.push(h2("2.5 Give Claude a Role（system prompt 设角色）"));
children.push(p("System prompt 哪怕只有一句「You are a helpful coding assistant specializing in Python」也能显著定调。这是「最低成本最高回报」的一招。"));

children.push(h2("2.6 Long Context Prompting（长文档处理三招）"));
children.push(bullet("① 长文档放在 Prompt 顶部，指令/问题/示例放后面——这一项调整在测试中最多能提升 30%"));
children.push(bullet("② 多文档用 <document><document_content/><source/></document> 嵌套结构"));
children.push(bullet("③ 让模型「先引用相关原文，再回答」——抑制噪声"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 3: Output Control =====
children.push(h1("三、输出与格式控制"));

children.push(h2("3.1 控制冗长度（Verbosity）"));
children.push(p("Claude 4.7 起会根据任务复杂度自动校准输出长度——简单查询变短、开放分析变长。如果要稳定风格："));
children.push(codeBlock(`Provide concise, focused responses. Skip non-essential
context, and keep examples minimal.`));
children.push(p("正面示例（「该这样写」）比负面禁令（「不要怎样」）更有效。"));

children.push(h2("3.2 控制 Markdown 与列表使用"));
children.push(p("这是新模型的常见痛点：默认 markdown 过多。官方推荐的「散文优先」prompt 段落："));
children.push(codeBlock(`<avoid_excessive_markdown_and_bullet_points>
When writing reports, documents, technical explanations,
analyses, or any long-form content, write in clear, flowing
prose using complete paragraphs and sentences.

DO NOT use ordered lists or unordered lists unless:
  a) you're presenting truly discrete items, or
  b) the user explicitly requests a list or ranking

Instead of listing items with bullets or numbers, incorporate
them naturally into sentences.
</avoid_excessive_markdown_and_bullet_points>`));

children.push(h2("3.3 控制响应格式的四种手法"));
children.push(bullet("说「该怎样」而不是「不要怎样」"));
children.push(bullet("用 XML 标签框住目标内容：<smoothly_flowing_prose>...</smoothly_flowing_prose>"));
children.push(bullet("匹配风格：Prompt 怎么写，输出就倾向于怎么写"));
children.push(bullet("给出具体的样式偏好，不要只提抽象要求"));

children.push(h2("3.4 LaTeX 与文档创建"));
children.push(p("新模型默认会用 LaTeX 表达数学公式。如果你的下游是纯文本场景，请在 Prompt 里明确禁用 LaTeX："));
children.push(codeBlock(`Format your response in plain text only. Do not use LaTeX,
MathJax, or any markup notation such as $ or \\frac{}{}.
Write all math expressions using standard text characters
(e.g., "/" for division, "*" for multiplication).`));

children.push(h2("3.5 Prefill 已废弃（4.6+）"));
children.push(callout(
  "重要变更",
  "从 Claude 4.6 开始，「最后一轮 assistant 消息预填」已不支持，会返回 400 错误。如果你旧代码里用了 prefill 来强制开头/格式/续写，必须改为：① 用 system prompt 描述格式 ② 用 XML 标签约束 ③ 让模型直接输出 JSON / 用 schema 强校验。",
  "FFE6E6", "C00000"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 4: Tool Use =====
children.push(h1("四、Tool Use（工具使用）"));

children.push(h2("4.1 让 Claude 真正去「执行」而不是「建议」"));
children.push(p("新模型对指令字面执行更严格——「能不能改一下」可能只得到建议而不是代码改动。要让它默认执行："));
children.push(codeBlock(`<default_to_action>
By default, implement changes rather than only suggesting them.
If the user's intent is unclear, infer the most useful likely
action and proceed, using tools to discover any missing details
instead of guessing.
</default_to_action>`));
children.push(p("反方向需要 Claude「先问再做」时："));
children.push(codeBlock(`<do_not_act_before_instructions>
Do not jump into implementation or change files unless clearly
instructed to make changes. When the user's intent is ambiguous,
default to providing information, doing research, and
providing recommendations rather than taking action.
</do_not_act_before_instructions>`));

children.push(h2("4.2 鼓励并行工具调用（提速利器）"));
children.push(p("最有效的 Prompt 之一，能把并行工具调用率拉到 ~100%："));
children.push(codeBlock(`<use_parallel_tool_calls>
If you intend to call multiple tools and there are no
dependencies between the tool calls, make all of the independent
tool calls in parallel.

For example, when reading 3 files, run 3 tool calls in parallel.

However, if some tool calls depend on previous calls to inform
dependent values, do NOT call these tools in parallel.
Never use placeholders or guess missing parameters.
</use_parallel_tool_calls>`));

children.push(h2("4.3 防止「过度激进」的反向调"));
children.push(callout(
  "新模型反直觉之处",
  "4.5/4.6 起对 system prompt 反应更强。若旧代码里有大量「CRITICAL: You MUST use this tool ...」式压力语言，新模型会过度触发工具。改为「Use this tool when ...」即可。",
  "EAF3FA", "2E75B6"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 5: Thinking & Reasoning =====
children.push(h1("五、Thinking 与 Reasoning"));

children.push(h2("5.1 Adaptive Thinking + Effort 双旋钮"));
children.push(p("旧模型用 thinking + budget_tokens 手动设思考预算；4.6+ 用 adaptive thinking（动态决定思不思）+ effort 参数（low/medium/high/xhigh/max）调强度。"));
children.push(makeTable(
  [1400, 7960],
  ["Effort 等级", "适用场景"],
  [
    ["max", "智力极端敏感任务；可能因「过思」反而下降，需 A/B 测"],
    ["xhigh（4.7 新）", "Coding / Agentic 默认推荐"],
    ["high", "智力敏感任务的最低门槛（推荐起点）"],
    ["medium", "成本敏感、token 受限"],
    ["low", "短任务 / 低智力要求 / 延迟敏感"],
  ]
));
children.push(callout(
  "工程提醒",
  "4.7 在 max / xhigh 时建议把 max_tokens 设到 64k 以上，给思考和 subagent 留余量。否则中途断流。",
  "FFF4CE", "F0C040"
));

children.push(h2("5.2 防止「想太多」"));
children.push(p("4.6 起有「过度探索」倾向。两手对策："));
children.push(bullet("从 prompt 里删掉旧时代的「If in doubt, use [tool]」式过激指令"));
children.push(bullet("降低 effort；或显式加约束："));
children.push(codeBlock(`Thinking adds latency and should only be used when it will
meaningfully improve answer quality — typically for problems
that require multi-step reasoning. When in doubt, respond directly.`));

children.push(h2("5.3 思考行为的四个微调技巧"));
children.push(bullet("「Think thoroughly」这种宽泛指令往往比人手写步骤更好"));
children.push(bullet("Few-shot 示例里用 <thinking>...</thinking> 演示推理风格，模型会模仿"));
children.push(bullet("无 thinking 模式下，加 <thinking>/<answer> 标签做手动 CoT"));
children.push(bullet("Self-check：「Before you finish, verify your answer against [criteria].」"));
children.push(callout(
  "小贴士",
  "Claude 4.5/4.6 在 thinking 关闭时对「think」一词高度敏感——想避开可改用「consider」「evaluate」「reason through」。",
  "EAF3FA", "2E75B6"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 6: Agentic Systems =====
children.push(h1("六、Agentic Systems（重头戏）"));

children.push(h2("6.1 长程推理与状态追踪"));
children.push(p("新模型擅长「跨多个 context window 接续工作」。关键工程技巧："));
children.push(bullet("结构化 state（如 tests.json）比自然语言 state 更稳"));
children.push(bullet("用 git 做检查点 / 回滚"));
children.push(bullet("进度笔记用自由文本即可"));
children.push(bullet("强调增量进度，反对一口气做完"));

children.push(h3("Context Awareness（Token Budget）"));
children.push(p("4.6 起模型能感知自己剩多少 context。若你的 harness 支持 context 压缩 / 写盘，告诉模型："));
children.push(codeBlock(`Your context window will be automatically compacted as it
approaches its limit, allowing you to continue working
indefinitely from where you left off. Therefore, do not stop
tasks early due to token budget concerns. As you approach
your limit, save your current progress and state to memory
before the context window refreshes.`));

children.push(h3("多 context window 工作流的 6 步法"));
children.push(bullet("首个 window 用专属 prompt 搭骨架（写 tests、init.sh）"));
children.push(bullet("结构化测试文件（tests.json），并明确不许删测试"));
children.push(bullet("提供 init.sh 类启动脚本，避免每次重启都重做配置"));
children.push(bullet("Context 用满后，开新 window 比 compact 更稳——靠 git/progress.txt 恢复状态"));
children.push(bullet("提供自动验证工具（Playwright/MCP/computer use）"));
children.push(bullet("鼓励一次性「用满」当前 context 而非提前收尾"));

children.push(h2("6.2 自主性 vs 安全性（关键 Prompt 之一）"));
children.push(p("4.6 默认会做一些「难撤销」的操作。务必加这段："));
children.push(codeBlock(`Consider the reversibility and potential impact of your actions.
You are encouraged to take local, reversible actions like
editing files or running tests, but for actions that are hard
to reverse, affect shared systems, or could be destructive,
ask the user before proceeding.

Examples that warrant confirmation:
- Destructive: deleting files, dropping tables, rm -rf
- Hard to reverse: git push --force, git reset --hard
- Visible to others: pushing code, commenting on PRs,
  sending messages, modifying shared infrastructure

When encountering obstacles, do not use destructive actions
as a shortcut.`));

children.push(h2("6.3 研究与信息收集"));
children.push(p("复杂研究任务推荐这段模板："));
children.push(codeBlock(`Search for this information in a structured way. As you gather
data, develop several competing hypotheses. Track your confidence
levels in your progress notes to improve calibration. Regularly
self-critique your approach and plan. Update a hypothesis tree
or research notes file to persist information and provide
transparency. Break down this complex research task systematically.`));

children.push(h2("6.4 Subagent 编排"));
children.push(p("4.6 会主动产生 subagent，但也容易「滥用」。控制其触发："));
children.push(codeBlock(`Use subagents when tasks can run in parallel, require isolated
context, or involve independent workstreams that don't need to
share state.

For simple tasks, sequential operations, single-file edits, or
tasks where you need to maintain context across steps, work
directly rather than delegating.`));

children.push(h2("6.5 高频常见问题（直接抄 prompt）"));

children.push(h3("(a) 反过度工程（Overeagerness）"));
children.push(codeBlock(`Avoid over-engineering. Only make changes that are directly
requested or clearly necessary. Keep solutions simple:

- Scope: Don't add features, refactor code, or make
  "improvements" beyond what was asked.
- Documentation: Don't add docstrings or comments to code
  you didn't change.
- Defensive coding: Don't add error handling or validation
  for scenarios that can't happen.
- Abstractions: Don't create helpers for one-time operations.

The right amount of complexity is the minimum needed.`));

children.push(h3("(b) 反「为了通过测试硬编码」"));
children.push(codeBlock(`Please write a high-quality, general-purpose solution. Do not
create helper scripts or workarounds. Implement a solution that
works correctly for all valid inputs, not just the test cases.
Tests verify correctness; they do not define the solution.

If the task is unreasonable or the tests are incorrect, inform
me rather than working around them.`));

children.push(h3("(c) 反幻觉（强制先读再答）"));
children.push(codeBlock(`<investigate_before_answering>
Never speculate about code you have not opened. If the user
references a specific file, you MUST read the file before
answering. Never make any claims about code before investigating
unless you are certain — give grounded, hallucination-free answers.
</investigate_before_answering>`));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 7: Behavior changes =====
children.push(h1("七、新一代模型的 6 个行为变化（必记）"));

children.push(makeTable(
  [2200, 7160],
  ["变化", "对你的 Prompt 意味着什么"],
  [
    ["更字面化的指令理解", "想让某指令应用到「所有项」就明说 scope，不要假设模型会泛化"],
    ["默认更少冗长", "想要详细，就显式要求；想要简洁，给正面示范"],
    ["Verbosity 自适应", "若产品依赖固定篇幅，需要 prompt 显式约束"],
    ["主动 subagent 倾向", "Prompt 里描述「何时该 / 不该」spawn subagent"],
    ["对工具更克制", "若 4.6 不调工具，加显式触发条件；不要再用「CRITICAL/MUST」高压语言"],
    ["前端默认审美固定（4.7）", "需要不同风格必须显式给出色板、字体、布局规格，否则总是同一风格"],
  ]
));

children.push(callout(
  "迁移到 4.6+ 的 6 步清单",
  [
    "1) 删掉旧时代的「CRITICAL / MUST」高压语；",
    "2) 删掉「If in doubt, use [tool]」式过激触发；",
    "3) 把 budget_tokens 换成 adaptive thinking + effort；",
    "4) 移除最后一轮的 prefilled assistant 消息；",
    "5) 显式描述 scope（不要依赖隐式泛化）；",
    "6) 加「reversibility & confirmation」安全 prompt。",
  ],
  "EAF3FA", "2E75B6"
));

// ===== Part 8: 实战映射 =====
children.push(h1("八、与「量化投资 Agent」的实战映射"));

children.push(makeTable(
  [2400, 6960],
  ["官方原则", "在你的量化 Agent 中的应用"],
  [
    ["金科玉律", "每个角色 Agent 的 system prompt 都让一个完全不懂量化的同事读——看得懂才合格"],
    ["Be Clear and Direct", "数据 Tool 描述要明示：「股票代码必须是 6 位数字 + .SH/.SZ 后缀」"],
    ["XML 结构化", "Supervisor 的 prompt 内：<task><context><available_agents><output_schema> 各一段"],
    ["示例少而精", "策略 DSL 提供 3-5 个示例（双均线/RSI/布林带），覆盖不同算子"],
    ["长文档放顶部", "RAG 研报放 system 顶部，问题放底部"],
    ["默认执行 vs 默认审慎", "回测 Agent 用 default_to_action；交易/下单 Agent 用 do_not_act_before_instructions"],
    ["并行工具调用", "数据拉取 5 只股票时强制并行；防止串行慢死"],
    ["Adaptive Thinking + Effort", "因子挖掘 / 策略反思用 high；批量摘要研报用 low"],
    ["Reversibility Prompt", "禁止 Agent 真的下单 / 调仓 / 删除回测记录——必须 ask first"],
    ["Self-check", "回测 Agent 在产报告前自检：未来函数？过拟合？换手率？"],
    ["Subagent 控制", "量化研究院 Supervisor 加明确「何时分派子 Agent」规则，避免一个简单查询拉起 5 个 subagent"],
    ["反 Overeagerness", "策略代码 Agent 加「Don't add features beyond what was asked」，避免乱加止损"],
  ]
));

children.push(callout(
  "如果你只能记 7 件事",
  [
    "1) 把 Claude 当聪明新员工——背景越足越好；",
    "2) 用 XML 标签结构化所有输入；",
    "3) 长文档放顶部，问题放底部；",
    "4) 示例 3-5 个，相关 + 多样 + 结构化；",
    "5) 想要执行就说「implement」，不要说「suggest」；",
    "6) 并行工具调用要显式 prompt；",
    "7) Adaptive Thinking + Effort 是新的「思考强度」总旋钮——旧的 budget_tokens 已淘汰。",
  ],
  "FFF4CE", "F0C040"
));

// ===== Part 9: KB integration =====
children.push(h1("九、写进个人知识库的卡片建议"));

children.push(h3("10_Concepts"));
children.push(bullet("Adaptive Thinking / Effort 参数"));
children.push(bullet("Context Awareness（token budget 感知）"));
children.push(bullet("Multishot / Few-shot Prompting"));
children.push(bullet("Prefill（已废弃，标注 deprecated）"));

children.push(h3("20_Patterns"));
children.push(bullet("XML-Structured Prompt 模板（instructions/context/examples/input/output_schema）"));
children.push(bullet("Long-Context Prompt 三招（顶部 + XML 文档 + 引用先行）"));
children.push(bullet("Parallel Tool Calls Prompt 模板"));
children.push(bullet("Reversibility & Confirmation Prompt 模板"));
children.push(bullet("Anti-Overeagerness / Anti-Hallucination Prompt 模板"));

children.push(h3("30_Frameworks"));
children.push(bullet("Claude API：thinking / effort / max_tokens 三参数搭配"));
children.push(bullet("Claude 4.6 / 4.7 迁移要点（6 步清单）"));

children.push(h3("50_Code"));
children.push(bullet("8 段可复用 system prompt 片段（本卡片 §4-§6 已收齐）"));

children.push(h3("70_Projects"));
children.push(bullet("量化 Agent 每个角色的 system prompt 模板"));

// ===== Build =====
const doc = new Document({
  creator: "Claude",
  title: "Anthropic Prompt Engineering 速通卡片",
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
        children: [tn("Anthropic Prompt Engineering · 速通卡片", { size: 18, color: "808080", italics: true })],
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
  const out = "/sessions/loving-elegant-archimedes/mnt/outputs/Anthropic_Prompt工程_速通卡片.docx";
  fs.writeFileSync(out, buffer);
  console.log("Wrote:", out, "size:", buffer.length);
});
