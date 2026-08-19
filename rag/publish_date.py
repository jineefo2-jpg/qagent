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
    m = _DATE_RE.search(name)
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    try:
        return _dt.date(y, mo, d).isoformat()
    except ValueError:
        return None


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
