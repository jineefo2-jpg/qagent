"""
结构化用户档案（Layer 1）。

字段（全部可选）：
  investment_style     "value" | "growth" | "balanced" | "momentum"
  risk_tolerance       "low" | "medium" | "high"
  markets              List[str]  ["US", "HK", "CN"]
  watchlist            List[str]  ["AAPL", "NVDA", ...]
  goals                str        "年化 20%" / "三年翻倍"
  background           str        "35 岁工程师，100w 可投资金"
  preferences          str        自由文本偏好（"偏好限价单，不喜欢期权"）
  blacklist_symbols    List[str]  Agent 不应推荐的标的
  custom               dict       未来扩展用

存储：Redis key = quant:profile:{user_id}
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from cache import cache


# 允许的标准字段（额外字段会塞到 custom）
ALLOWED_FIELDS = {
    "investment_style",
    "risk_tolerance",
    "markets",
    "watchlist",
    "goals",
    "background",
    "preferences",
    "blacklist_symbols",
}

# 取值校验
ENUM_STYLE = {"value", "growth", "balanced", "momentum"}
ENUM_RISK = {"low", "medium", "high"}
ENUM_MARKET = {"US", "HK", "CN", "JP", "SG"}

# 最大字段长度（防 LLM 写超长文本污染）
MAX_STRING_LEN = 500
MAX_LIST_LEN = 50


@dataclass
class UserProfile:
    user_id: str
    investment_style: Optional[str] = None
    risk_tolerance: Optional[str] = None
    markets: List[str] = field(default_factory=list)
    watchlist: List[str] = field(default_factory=list)
    goals: Optional[str] = None
    background: Optional[str] = None
    preferences: Optional[str] = None
    blacklist_symbols: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        # 兼容老数据缺字段
        return cls(
            user_id=d.get("user_id", ""),
            investment_style=d.get("investment_style"),
            risk_tolerance=d.get("risk_tolerance"),
            markets=list(d.get("markets") or []),
            watchlist=list(d.get("watchlist") or []),
            goals=d.get("goals"),
            background=d.get("background"),
            preferences=d.get("preferences"),
            blacklist_symbols=list(d.get("blacklist_symbols") or []),
            custom=dict(d.get("custom") or {}),
            created_at=float(d.get("created_at", 0)),
            updated_at=float(d.get("updated_at", 0)),
        )


def _key(user_id: str) -> str:
    return f"quant:profile:{user_id}"


def get_profile(user_id: str) -> UserProfile:
    """读档；不存在则返回空白 profile（不持久化）"""
    if not user_id:
        return UserProfile(user_id="")
    d = cache.get(_key(user_id))
    if not d:
        return UserProfile(user_id=user_id)
    return UserProfile.from_dict(d)


def _sanitize_value(field_name: str, value: Any) -> Any:
    """按字段类型做基本校验/截断，防 LLM 乱写"""
    if value is None:
        return None
    if field_name == "investment_style":
        v = str(value).strip().lower()
        return v if v in ENUM_STYLE else None
    if field_name == "risk_tolerance":
        v = str(value).strip().lower()
        return v if v in ENUM_RISK else None
    if field_name == "markets":
        if not isinstance(value, list):
            return None
        return [m for m in (str(x).strip().upper() for x in value)
                if m in ENUM_MARKET][:MAX_LIST_LEN]
    if field_name in ("watchlist", "blacklist_symbols"):
        if not isinstance(value, list):
            return None
        cleaned = []
        for s in value:
            s2 = str(s).strip().upper()
            if 1 <= len(s2) <= 10:  # ticker 长度限制
                cleaned.append(s2)
        return list(dict.fromkeys(cleaned))[:MAX_LIST_LEN]  # 去重保序
    if field_name in ("goals", "background", "preferences"):
        return str(value).strip()[:MAX_STRING_LEN] or None
    return None


def update_profile_fields(user_id: str, updates: Dict[str, Any]) -> UserProfile:
    """
    部分更新：传 {field: new_value} 字典。
    传 None 等于"清空该字段"。
    未识别字段塞进 custom（隔离，不污染主 schema）。
    """
    if not user_id:
        raise ValueError("user_id 不能为空")

    profile = get_profile(user_id)
    if profile.created_at == 0:
        profile.created_at = time.time()

    for k, v in (updates or {}).items():
        if k in ALLOWED_FIELDS:
            cleaned = _sanitize_value(k, v)
            # 显式 None 表示清空
            if v is None:
                if k in ("markets", "watchlist", "blacklist_symbols"):
                    setattr(profile, k, [])
                else:
                    setattr(profile, k, None)
            elif cleaned is not None or k in ("markets", "watchlist", "blacklist_symbols"):
                setattr(profile, k, cleaned)
        else:
            # 不认识的字段进 custom；做长度限制
            key = str(k)[:50]
            val_str = str(v)[:MAX_STRING_LEN]
            profile.custom[key] = val_str

    profile.updated_at = time.time()
    cache.set(_key(user_id), profile.to_dict(), ttl=None)
    return profile


def clear_profile(user_id: str) -> None:
    cache.delete(_key(user_id))


def profile_summary_text(profile: UserProfile, max_lines: int = 8) -> str:
    """
    生成给 LLM system prompt 用的紧凑摘要。
    返回空字符串如果 profile 完全空白（不污染 prompt）。
    """
    lines = []
    if profile.investment_style:
        lines.append(f"- 投资风格: {profile.investment_style}")
    if profile.risk_tolerance:
        lines.append(f"- 风险偏好: {profile.risk_tolerance}")
    if profile.markets:
        lines.append(f"- 关注市场: {', '.join(profile.markets)}")
    if profile.watchlist:
        wl = profile.watchlist[:15]
        suffix = f" 等 {len(profile.watchlist)} 个" if len(profile.watchlist) > 15 else ""
        lines.append(f"- 自选股: {', '.join(wl)}{suffix}")
    if profile.goals:
        lines.append(f"- 投资目标: {profile.goals}")
    if profile.background:
        lines.append(f"- 个人背景: {profile.background}")
    if profile.preferences:
        lines.append(f"- 偏好: {profile.preferences}")
    if profile.blacklist_symbols:
        lines.append(f"- 不喜欢的标的: {', '.join(profile.blacklist_symbols[:10])}")
    if profile.custom:
        items = list(profile.custom.items())[:3]
        kv = "; ".join(f"{k}={v}" for k, v in items)
        lines.append(f"- 其他: {kv}")

    if not lines:
        return ""

    return "【用户档案】\n" + "\n".join(lines[:max_lines])
