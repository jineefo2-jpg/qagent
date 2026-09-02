// Career planning addendum: 量化系统工程 × 金融 AI 大模型
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
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
  children: [tn("职业方向规划：量化系统工程 × 金融 AI 大模型", { size: 42, bold: true, color: "1F4E79" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [tn("学习路线的配套职业地图 · 12-24 个月行动方案", { size: 26, color: "2E75B6" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 360 },
  children: [tn("（配合主文档《Agent 学习与训练路线》与《12 周执行手册》使用）", { size: 22, italics: true, color: "808080" })],
}));

children.push(callout(
  "一句话定位",
  "你处在一个稀缺的「窗口期」——传统量化偏底层系统、传统金融 AI 偏建模，而「Agent 工程 + 量化领域知识」复合人才正在被高薪争抢。本规划帮你押注这个交叉点。",
  "EAF3FA", "2E75B6"
));

children.push(callout(
  "免责声明",
  "薪资数据来自 2024-2025 年公开信息（脉脉、Levels.fyi、Glassdoor、招聘网站、行业访谈）。具体公司、年份、地区差异极大，仅供参考量级。求职前务必交叉验证。",
  "FFE6E6", "C00000"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 1: 两个方向定位 =====
children.push(h1("一、两个方向：横向对比"));

children.push(makeTable(
  [1800, 3760, 3800],
  ["维度", "量化系统工程", "金融 AI 大模型"],
  [
    ["核心定位", "为量化策略服务的「水电煤」：行情、撮合、回测、风控、低延迟交易系统", "用 LLM / Agent 改造金融业务：研报分析、投顾、客服、合规、量化研究助手"],
    ["代表问题", "把策略 RT 从 100µs 降到 10µs；构建跨市场行情归一化；做百亿级回测平台", "造一个能读 1000 份研报输出观点的 Agent；让 LLM 不在合规话术上出错；多 Agent 投研工作流"],
    ["技术栈骨架", "C++ / Rust / Python / 分布式系统 / KDB+/q / TimescaleDB / FPGA / 内核网络", "Python / LLM API / RAG / 向量库 / LangGraph / Claude Agent SDK / MCP / 评估框架"],
    ["金融领域要求", "深：必须懂市场结构、订单簿、撮合规则、风险控制", "中：懂业务术语、报表、监管框架即可，重在 LLM 工程能力"],
    ["成熟度", "成熟行业，门槛高、范式稳定；竞争激烈但路径清晰", "新兴方向，标准未定；机会多但项目可能死掉"],
    ["薪资天花板（国内）", "高（顶级私募 PM 路径可超千万）", "中高（资深 AI 工程师 100-200w 居多）"],
    ["薪资天花板（海外）", "极高（HFT 资深可超 USD 1M）", "高（FAANG/银行 AI lead 300-600k USD）"],
    ["职业稳定性", "高（行业不轻易消失，资深人才稀缺）", "中（产品/创业波动大，但能力可迁移）"],
    ["复合优势", "+AI 能力会让你在量化里「鹤立鸡群」", "+量化背景会让你在金融 AI 圈极其稀缺"],
  ]
));

children.push(callout(
  "最关键的判断",
  [
    "纯量化系统工程——卷得厉害，但护城河深。",
    "纯金融 AI 大模型——风口正盛，但项目死亡率高。",
    "「Agent 工程 + 量化领域」复合人才——是当前最稀缺、议价能力最强的位置。这正是你 12 周学习路线在押注的方向。",
  ],
  "FFF4CE", "F0C040"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 2: 量化系统工程详解 =====
children.push(h1("二、量化系统工程方向"));

children.push(h2("2.1 行业地图"));
children.push(p("国内量化行业分四类玩家，对系统工程师的需求与文化各异："));
children.push(makeTable(
  [1800, 3360, 4200],
  ["玩家类型", "代表机构", "技术与文化"],
  [
    ["头部私募", "幻方、九坤、明汯、灵均、宽德、衍复", "Citadel/Two Sigma 风格，技术驱动，薪资极高，但 996 + 强 KPI"],
    ["新兴私募", "稳博、思勰、洛书、玖瀛、星阔", "更年轻、灵活，给得起股权；技术债少"],
    ["券商自营", "中信、华泰、国君、招商资管", "稳定、规范、待遇中等；适合做 to B 系统"],
    ["海外/合资", "Two Sigma 上海、Optiver、Jane Street（少）、Citadel 中国", "薪资标杆、文化国际化、面试极难（≥ 6 轮编程）"],
  ]
));

children.push(h2("2.2 典型岗位"));
children.push(makeTable(
  [2200, 3600, 3560],
  ["岗位", "干什么", "典型技术栈"],
  [
    ["低延迟交易系统工程师", "撮合接入、策略下单链路、内核绕过网络", "C++17/20、DPDK、Solarflare、FPGA、内核调优"],
    ["行情系统工程师", "Tick 数据接收/归一化/分发；存储压缩", "C++ / Rust、KDB+/q、ClickHouse、TimescaleDB"],
    ["回测平台工程师", "构建百亿级矢量化回测引擎；分布式调度", "Python+C++ 混合、Polars、Ray、Dask"],
    ["策略基础设施工程师", "因子计算管道、Pipeline、调度", "Pandas/Polars、Airflow/Prefect、DuckDB"],
    ["风控系统工程师", "事前/事中/事后风控；实时持仓、暴露", "C++ / Java、Kafka、Redis、规则引擎"],
    ["量化研究员（研究侧）", "因子挖掘、策略开发、组合优化", "Python、统计、ML/DL、QLib"],
    ["量化系统 SRE", "集群、监控、容灾、性能调优", "K8s、Prometheus、eBPF、Linux 内核"],
  ]
));

children.push(h2("2.3 核心能力地图"));
children.push(h3("硬技能（必须）"));
children.push(bullet("一门系统级语言：C++（首选）/ Rust / Go（仅部分公司接受）"));
children.push(bullet("Python 工程化（不止脚本水平）：异步、性能、打包"));
children.push(bullet("数据结构与算法：能现场写红黑树、跳表、LRU、Top-K"));
children.push(bullet("Linux 与网络：epoll/io_uring、TCP 调优、零拷贝"));
children.push(bullet("分布式系统：一致性、消息队列、时序数据库"));
children.push(bullet("市场微结构：订单簿、撮合规则、做市策略基础"));
children.push(h3("软技能（区分中高级）"));
children.push(bullet("能与研究员对齐需求并把模糊想法变成系统设计"));
children.push(bullet("性能直觉：能在白板上估算延迟瓶颈与 throughput 上限"));
children.push(bullet("稳健性意识：每一行代码都问「如果挂了影响多少钱」"));
children.push(bullet("英语阅读：核心论文与开源代码绝大多数是英文"));

children.push(h2("2.4 薪资与晋升（2024-2025 参考）"));
children.push(callout(
  "数据来源说明",
  "下表为国内一二线城市行情综合估计，单位为人民币 / 年（含奖金，不含期权/Carry）。海外薪资差异极大，仅给区间。",
  "EAF3FA", "2E75B6"
));
children.push(makeTable(
  [1800, 2200, 2200, 3160],
  ["职级", "国内头部私募", "国内券商/中型", "海外（USD）"],
  [
    ["应届/初级", "60-150w", "30-60w", "180-300k"],
    ["3-5 年", "150-300w", "60-120w", "300-500k"],
    ["资深 / Lead", "300-600w", "120-250w", "500-800k"],
    ["顶级技术专家", "500w-千万+", "200-400w", "800k-1.5M"],
    ["PM / 合伙人", "千万级", "200-500w + 分红", "1M+"],
  ]
));
children.push(callout(
  "提醒",
  [
    "国内数字含奖金浮动巨大——基础工资可能 30-50w，剩下全靠业绩奖。",
    "Carry / 利润分成才是高级岗位真正的天花板，但需要 5+ 年绑定。",
    "「offer 表」上的天文数字往往是 outlier，要看 median 不要看 top。",
  ],
  "FFF4CE", "F0C040"
));

children.push(h2("2.5 你目前的进入难点与对策"));
children.push(richBullet([tn("难点 1：", { bold: true }), tn("C++ 基础——量化系统工程师的入场券。建议在 12 周学习路线之外，单开「每天 1 小时 C++」长期任务。")]));
children.push(richBullet([tn("难点 2：", { bold: true }), tn("市场微结构——靠书 + 实战。推荐《算法交易（Algorithmic Trading）》Chan、《Trading and Exchanges》Harris、上交所/深交所交易规则原文。")]));
children.push(richBullet([tn("难点 3：", { bold: true }), tn("无量化背景的简历——靠开源项目 + Kaggle/聚宽/Ricequant 实盘记录。")]));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 3: 金融 AI 大模型详解 =====
children.push(h1("三、金融 AI 大模型方向"));

children.push(h2("3.1 行业地图"));
children.push(p("金融 AI 大模型有四类雇主，技术深度与业务场景差异巨大："));
children.push(makeTable(
  [1800, 3360, 4200],
  ["玩家类型", "代表机构 / 产品", "技术与文化"],
  [
    ["大型银行 AI", "工行、招行、平安、JPMorgan IndexGPT、Morgan Stanley AI@MS", "规模大、合规重、技术保守；适合做 to B 工作流落地"],
    ["券商/资管 AI", "华泰、国君、易方达、贝莱德 Aladdin Copilot", "投研助手、研报生成、合规问答是主战场"],
    ["金融科技/独立 AI 厂商", "彭博 BloombergGPT、Hebbia、Kensho、度小满轩辕、蚂蚁金融大模型", "技术激进、产品化要求高；薪资和股权诱人但波动大"],
    ["传统科技大厂金融部门", "百度 / 字节 / 阿里的金融 NLP 团队、Microsoft 的金融 Copilot", "工程能力强、平台资源好；偏 to B 大客户"],
  ]
));

children.push(h2("3.2 典型岗位"));
children.push(makeTable(
  [2200, 3600, 3560],
  ["岗位", "干什么", "典型技术栈"],
  [
    ["LLM 应用工程师（金融）", "RAG / Agent / 投研助手开发", "LangGraph / Claude Agent SDK / 向量库 / Eval"],
    ["金融领域 NLP 研究员", "金融语料微调、Embedding 训练", "Llama/Qwen 微调、LoRA、领域语料"],
    ["AI 产品工程师", "把 LLM 能力封装成可用产品", "全栈 + LLM + 评估闭环"],
    ["AI 评估 / 安全工程师", "合规、防幻觉、Eval 基线建设", "RAGAS / DeepEval / 红队测试"],
    ["量化研究 × AI 复合", "用 LLM 辅助因子挖掘、研报抽取", "本路线主线"],
    ["AI 解决方案架构师", "to B 客户场景设计与落地", "顾问 + 工程能力 + 金融业务"],
  ]
));

children.push(h2("3.3 核心能力地图"));
children.push(h3("硬技能"));
children.push(bullet("LLM 工程闭环：Prompt / RAG / Tool Use / Agent / Eval（你的 12 周路线全覆盖）"));
children.push(bullet("Python 工程化：FastAPI / 异步 / 测试 / 部署"));
children.push(bullet("向量数据库与检索：Chroma → Qdrant → Milvus，懂混合检索"));
children.push(bullet("基础 ML/DL：能看懂论文、调通 fine-tune 流程（不要求做研究）"));
children.push(bullet("金融业务理解：财务三张表、监管框架、合规红线、研报结构"));
children.push(bullet("评估与可观测：建评估集、跑 Langfuse / Phoenix"));

children.push(h3("软技能（决定职业高度）"));
children.push(bullet("把模糊业务需求翻译成 LLM 可解的形态"));
children.push(bullet("能给非技术金融人员讲清模型边界与不可承诺事项"));
children.push(bullet("项目落地能力——LLM 项目死于「demo 好看但用不上」的太多"));
children.push(bullet("跨团队协作：与业务、合规、风控、IT 反复打磨流程"));

children.push(h2("3.4 薪资与晋升（2024-2025 参考）"));
children.push(makeTable(
  [1800, 2200, 2200, 3160],
  ["职级", "国内金融科技", "国内银行/券商", "海外（USD）"],
  [
    ["应届/初级", "30-60w", "20-40w", "150-250k"],
    ["3-5 年", "60-150w", "40-100w", "250-400k"],
    ["资深 / Lead", "150-300w", "100-200w", "400-600k"],
    ["技术专家 / 架构师", "300-500w", "200-400w", "600-900k"],
    ["AI 负责人 / CTO 路径", "500w + 期权", "300-600w", "900k+ + 期权"],
  ]
));
children.push(callout(
  "现实考量",
  [
    "彭博、Hebbia、Kensho 这类「AI + 金融数据」厂商对中国大陆候选人开放度有限。",
    "国内大行 AI 部门薪资天花板更低但稳定；金融科技公司薪资高但项目寿命短。",
    "「Agent 工程能力 + 量化领域知识」组合，能让你跳过初级阶段，直接谈资深包。",
  ],
  "FFF4CE", "F0C040"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 4: 复合人才路线 =====
children.push(h1("四、最优解：「Agent 工程 + 量化领域」复合路径"));

children.push(p("两个方向不必二选一。事实上，目前市场上最稀缺、议价能力最强的人才画像是："));
children.push(callout(
  "目标画像",
  [
    "「能用 Agent 工程做出量化研究助手 / 因子挖掘助手 / 投研工作流的复合型工程师」",
    "",
    "向量化机构看，你不是「写 C++ 的」而是「能放大每个研究员产能的人」；",
    "向金融科技公司看，你不是「LLM 工程师」而是「懂量化业务的 LLM 专家」。",
    "两边都因为「另一边的能力」溢价。",
  ],
  "EAF3FA", "2E75B6"
));

children.push(h2("4.1 这个画像的市场证据"));
children.push(bullet("幻方、九坤等头部私募近 1 年起设「AI 研究助手」相关 JD，要求 LLM/Agent 经验 + 量化业务理解"));
children.push(bullet("彭博 / Kensho / Hebbia 高薪招「Quant Background + LLM」combo"));
children.push(bullet("券商 AI 实验室（华泰、招商、国君）专设「投研 AI 工程师」岗位"));
children.push(bullet("AI 创业公司的金融 vertical 几乎都缺懂业务的算法/工程同事"));

children.push(h2("4.2 4 个具体细分定位（任选 1-2 个深扎）"));
children.push(makeTable(
  [2400, 6960],
  ["定位", "你做什么 / 你卖什么"],
  [
    ["量化研究助手工程师", "在量化私募做内部产品：让研究员用对话/Agent 完成 80% 重复工作"],
    ["投研 Agent 产品工程师", "在券商/资管做对客 toC/toB 投研助手"],
    ["金融数据 Agent 平台工程师", "做研报/财务/舆情的 RAG + Agent 平台，类似国内版 Hebbia"],
    ["AI 评估 / 风控工程师", "专做金融 LLM 的 Eval / 合规 / 防幻觉，相对蓝海"],
  ]
));

children.push(h2("4.3 简历叙事三段法（基于 12 周项目）"));
children.push(p("当 W12 结束你拥有「完整量化研究院多 Agent 系统」时，简历可以这样讲："));
children.push(richBullet([tn("段一 · 系统：", { bold: true }), tn("「设计并实现 6 角色多 Agent 量化研究系统，端到端任务完成率 ≥ 70%，单次成本 < $1」")]));
children.push(richBullet([tn("段二 · 工程：", { bold: true }), tn("「自建 Factor DSL + 沙箱执行 + Reflexion 循环，自动挖掘因子并去伪存真」")]));
children.push(richBullet([tn("段三 · 复用：", { bold: true }), tn("「沉淀通用 Agent 框架已迁移到 X / Y / Z 领域（哪怕只是 1-2 周的小迁移），证明能力可平移」")]));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 5: 12-24 个月行动计划 =====
children.push(h1("五、12-24 个月行动计划"));

children.push(h2("5.1 第 0-3 个月：与主路线对齐"));
children.push(p("严格按 12 周执行手册推进，**不再额外加新项目**。本阶段唯一职业准备动作："));
children.push(bullet("把 GitHub 整理好：README 写明意图与亮点，避免「学习仓库」式堆代码"));
children.push(bullet("注册 LinkedIn、脉脉、推特/X，关注量化与 AI 头部账号建立信息源"));
children.push(bullet("初步选定 4-8 家目标公司，研读其 JD、产品、公开技术博客"));

children.push(h2("5.2 第 4-6 个月：补短板 + 公开输出"));
children.push(bullet("如果方向偏「量化系统工程」：每天 1 小时 C++，目标做出 1 个低延迟项目（如撮合引擎 toy）"));
children.push(bullet("如果方向偏「金融 AI」：把 12 周项目升级为对外可用的产品页（部署 + 演示 + 用户反馈）"));
children.push(bullet("发布 3-5 篇技术博客，至少 1 篇是「Agent + 量化」的独立观点"));
children.push(bullet("参加 1-2 次行业 meetup / 线上 talk，建立第一批弱关系"));

children.push(h2("5.3 第 7-9 个月：精修一个高难度项目"));
children.push(p("选一个能在面试中讲 30 分钟的旗舰项目深入打磨："));
children.push(bullet("方向 A：基于 LLM 的实盘策略助手（接通真实小账户做纸交易）"));
children.push(bullet("方向 B：因子挖掘 Agent 上 Kaggle / Numerai 公开比赛验证效果"));
children.push(bullet("方向 C：金融领域评估基准（如 50-100 题专家集 + 公开榜单）"));
children.push(callout(
  "项目质量 > 项目数量",
  "面试官最怕「20 个半成品」。一个能讲清「问题 → 设计 → 取舍 → 数据 → 结论」的大项目 ≥ 5 个 toy demo。",
  "EAF3FA", "2E75B6"
));

children.push(h2("5.4 第 10-12 个月：求职启动"));
children.push(bullet("做面试题专项：LeetCode 中等 200 题（量化方向加 hard 数学题）+ 系统设计 20 题 + Agent 工程题 20 题"));
children.push(bullet("做 5 次模拟面试（线上有偿服务或同行互助）"));
children.push(bullet("先投 3-5 家不那么向往的，把面试节奏 / 自我介绍打磨好"));
children.push(bullet("再投核心目标公司"));
children.push(bullet("Offer 谈判：拿到至少 2 个 offer 再做决定；薪资按 base + bonus + 期权三段比较"));

children.push(h2("5.5 第 13-24 个月：在岗位上加速"));
children.push(bullet("入职后 6 个月做出可见的小成绩（先抓 1 个高 ROI 的问题）"));
children.push(bullet("保持公开输出（即使脱敏后写），形成行业声誉"));
children.push(bullet("12-18 个月评估：是否要内部转岗到更核心团队 / 跳槽 / 拿期权 / 创业"));

children.push(callout(
  "求职时间线提醒",
  "国内金融/量化校招与社招节奏不同：校招集中在 9-11 月与 3-5 月；社招全年但年底奖金前慢、年初奖金后快。规划求职启动月份要看你的可面试节奏。",
  "FFF4CE", "F0C040"
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 6: 资源清单 =====
children.push(h1("六、补充资源清单（在主文档之外）"));

children.push(h2("6.1 量化系统工程必读"));
children.push(bullet("《Trading and Exchanges》Larry Harris—市场微结构圣经"));
children.push(bullet("《Algorithmic Trading》Ernest Chan—入门级实战"));
children.push(bullet("《C++ High Performance》Andrist & Sehr—性能向 C++"));
children.push(bullet("《Designing Data-Intensive Applications》Kleppmann—分布式与时序"));
children.push(bullet("LMAX Disruptor 论文 + 源码—金融级低延迟队列"));
children.push(bullet("Jane Street 技术博客 / Optiver 博客 / Two Sigma 工程博客"));

children.push(h2("6.2 金融 AI 大模型必读"));
children.push(bullet("BloombergGPT 论文（2023）—金融大模型早期范本"));
children.push(bullet("FinGPT / FinRobot / FinBen 项目—开源金融 Agent"));
children.push(bullet("Morgan Stanley AI@MS 公开案例—大行落地实战"));
children.push(bullet("Hebbia / Kensho 公开 demo 与产品页—对标"));
children.push(bullet("Anthropic《Building Effective Agents》—工程范式"));
children.push(bullet("Eugene Yan《Patterns for Building LLM-based Systems》—生产化"));

children.push(h2("6.3 求职信息渠道"));
children.push(bullet("脉脉 / LinkedIn / Levels.fyi—薪资与公司口碑"));
children.push(bullet("rebanker / 量职 / 量化 LP 公众号—量化岗位精选"));
children.push(bullet("华尔街见闻 / 财联社—行业动态"));
children.push(bullet("Hacker News / Reddit r/quant—海外信息源"));

children.push(h2("6.4 社区与同行"));
children.push(bullet("聚宽 / Ricequant / 优矿—量化策略实盘+社区"));
children.push(bullet("Discord 上的 LangChain / LlamaIndex / Anthropic 官方频道"));
children.push(bullet("机器之心 / 量子位 / The Decoder—AI 行业资讯"));
children.push(bullet("微信公众号「ChatGPT 实验室」「数据科学家联盟」「量化投资与机器学习」"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== Part 7: 风险与判断 =====
children.push(h1("七、风险预警与判断框架"));

children.push(h2("7.1 行业风险"));
children.push(makeTable(
  [2200, 7160],
  ["风险点", "对策"],
  [
    ["量化策略 alpha 衰减", "选偏「平台基础设施」方向，价值与个体策略表现解绑"],
    ["金融 AI 监管收紧", "选合规友好场景（投研助手、内部工作流），避开「AI 推荐买卖」"],
    ["LLM 价格继续下降", "好事：降低产品成本；坏事：纯封装 LLM 的小公司没护城河"],
    ["开源模型追平闭源", "本地化部署能力会越来越值钱，要保留这条技能"],
    ["AI 泡沫破裂", "锚定「能产生现金流的真实业务场景」，避免纯概念项目"],
  ]
));

children.push(h2("7.2 个人风险"));
children.push(bullet("过度押宝单一公司或单一技术栈——保持 6 个月内可跳槽的能力"));
children.push(bullet("以为「项目复杂 = 简历亮眼」——面试官只关心你解决了什么真实问题"));
children.push(bullet("忽视基础（DSA / 系统设计）——LLM 时代基础依然是面试主战场"));
children.push(bullet("低估软技能——「能讲清楚」决定 senior 以上岗位的天花板"));

children.push(h2("7.3 决策框架（每 6 个月自检）"));
children.push(p("每半年做一次「四象限自检」——把方向坚持下去 vs 调整方向："));
children.push(makeTable(
  [2400, 6960],
  ["问题", "判断依据"],
  [
    ["我的技能在这半年是否变得更值钱？", "对照岗位 JD 看自己缺什么，再看自己补上了什么"],
    ["市场对这个方向是否依然在加价？", "看头部公司 JD 数量、薪资中位数、融资动态"],
    ["我能否在专业人士面前讲清自己做的事？", "找 2 个比你资深的同行做 30 分钟讲解，观察反馈"],
    ["现在的工作 / 项目离我目标画像更近还是更远？", "如果连续 2 个 6 个月都更远——必须调整"],
  ]
));

children.push(callout(
  "写在最后",
  [
    "「量化系统工程 + 金融 AI 大模型」是非常好的双轨方向——前者给你护城河，后者给你想象力。",
    "12 周学习路线 + 这份职业规划组合在一起，是把「会用 LLM 的人」推进到「懂业务的资深 AI 工程师」的最短可行路径。",
    "其余的就是时间和耐心——这个赛道在 2026-2028 年都还有窗口，别急。",
  ],
  "EAF3FA", "2E75B6"
));

// ===== Build =====
const doc = new Document({
  creator: "Claude",
  title: "职业方向规划：量化系统工程 × 金融 AI 大模型",
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
        children: [tn("职业方向规划 · 量化系统工程 × 金融 AI 大模型", { size: 18, color: "808080", italics: true })],
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
  const out = "/sessions/loving-elegant-archimedes/mnt/outputs/职业方向规划_量化系统工程_金融AI大模型.docx";
  fs.writeFileSync(out, buffer);
  console.log("Wrote:", out, "size:", buffer.length);
});
