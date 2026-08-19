"""研报/公告 chunk 的 publish_date 解析（设计规格 §7 / 架构 A6）。

规则（不猜）：
  1. 文件名含 YYYY-MM-DD / YYYYMMDD / YYYY_MM_DD → 该日期，source='filename'
  2. 同目录 publish_dates.json 有 {"<文件名>": "YYYY-MM-DD"} → 覆盖，source='override'
  3. 都没有 → publish_date=""（chroma metadata 不接受 None），source='unknown'
     未打日期的 chunk 在回测/复盘场景中直接排除，不做兜底猜测。

入库时写入几乎零成本；事后回补需重新解析全部语料 —— 这一项不可延后。
检索侧的 as_of_date 过滤本期不做（P4）。
"""
from __future__ import annotations
import datetime as _dt
import json
import re
from pathlib import Path

_DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)")
OVERRIDE_FILE = "publish_dates.json"


def _from_filename(name: str) -> str | None:
    """文件名里有多个日期（如 `600519_2023-12-31_年报_发布2024-03-28.pdf`：报告期 + 发布日）→ 取【最大】。
    取最早会让年报提前三个月可见 —— PIT 不安全方向；取最晚至多损失召回。
    已知误判：B 股代码 + 两位后缀（`万科B_200002_01.pdf` → 2000-02-01），用同目录 publish_dates.json 覆盖。"""
    found: list[_dt.date] = []
    for m in _DATE_RE.finditer(name):
        y, mo, d = (int(x) for x in m.groups())
        try:
            found.append(_dt.date(y, mo, d))
        except ValueError:
            continue
    return max(found).isoformat() if found else None


def _from_override(pdf_path: Path) -> str | None:
    f = pdf_path.parent / OVERRIDE_FILE
    if not f.exists():
        return None
    try:
        table = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    v = table.get(pdf_path.name) or table.get(pdf_path.stem)
    if not v:
        return None
    try:
        return _dt.date.fromisoformat(str(v)).isoformat()
    except ValueError:
        return None


def resolve_publish_date(pdf_path: Path) -> tuple[str, str]:
    """→ (publish_date 'YYYY-MM-DD' 或 ''，source ∈ {'override', 'filename', 'unknown'})。override 优先于文件名。"""
    ov = _from_override(pdf_path)
    if ov:
        return ov, "override"
    fn = _from_filename(pdf_path.name)
    if fn:
        return fn, "filename"
    return "", "unknown"
