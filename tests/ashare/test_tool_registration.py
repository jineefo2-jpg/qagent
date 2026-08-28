"""A 股只读工具注册的四道防线（架构 §6.3 M1/M2/M4 + L4 import 禁令）。

守的是一件事：这些工具永远只读、永远对匿名可见、永远不被误加进 TRADING_TOOLS。
TRADING_TOOLS 的语义是「需要用户身份」而非「危险」—— 全市场公开数据没有用户维度。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

pytest.importorskip("duckdb")
from ashare.agent_tools import (ASHARE_READONLY_TOOLS, ASHARE_TOOL_REGISTRY,
                                ASHARE_TOOL_SCHEMAS)

qa = pytest.importorskip("quant_agent")


def test_m1_readonly_tools_never_in_trading_tools():
    assert ASHARE_READONLY_TOOLS.isdisjoint(qa.TRADING_TOOLS), (
        f"A 股只读工具被加进了 TRADING_TOOLS：{ASHARE_READONLY_TOOLS & qa.TRADING_TOOLS}")


def test_m2_anonymous_users_see_the_tools():
    anon = {s["name"] for s in qa.get_tool_schemas_for(False)}
    assert ASHARE_READONLY_TOOLS <= anon, (
        f"匿名用户看不到 {ASHARE_READONLY_TOOLS - anon} —— 多半是被误加进了 TRADING_TOOLS")


def test_registry_and_schemas_actually_registered():
    """extend 必须发生在 _OPENAI_TOOLS 固化之前 —— 固化后的 OpenAI 清单里必须有这两个工具。"""
    for name in ASHARE_READONLY_TOOLS:
        assert name in qa.TOOL_REGISTRY
    schema_names = {s["name"] for s in qa.TOOL_SCHEMAS}
    assert ASHARE_READONLY_TOOLS <= schema_names
    openai_names = {t["function"]["name"] for t in qa._OPENAI_TOOLS}
    assert ASHARE_READONLY_TOOLS <= openai_names, (
        "TOOL_SCHEMAS 里有、_OPENAI_TOOLS 里没有 = extend 放到了固化之后，新工具静默失效")


def test_m4_descriptions_are_readonly_and_not_persuasive():
    verbs = ("下单", "买入", "卖出", "委托")
    for s in ASHARE_TOOL_SCHEMAS:
        assert "只读" in s["description"], f"{s['name']} 描述缺「只读」声明"
        hits = [v for v in verbs if v in s["description"].replace("不下单", "")]
        assert not hits, f"{s['name']} 描述含交易动词 {hits}"


def test_l4_no_write_side_imports():
    """agent_tools 只许 import query / factors 读侧 / backtest.{store,metrics}。"""
    src = pathlib.Path("ashare/agent_tools.py").read_text(encoding="utf-8")
    banned = ("brokers", "ashare.data.ingest", "ashare.data.pipeline",
              "ashare.data.promote", "ashare.data.derived_store")
    for node in ast.walk(ast.parse(src)):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        for m in mods:
            assert not any(m == b or m.startswith(b + ".") for b in banned), \
                f"agent_tools import 了写侧/越权模块 {m}"


def test_tools_return_dicts_never_raise():
    """T1：任何输入（含越界日期）都返回 dict，绝不跨 LLM 边界抛异常。"""
    r = ASHARE_TOOL_REGISTRY["query_universe"]("2099-01-01")
    assert isinstance(r, dict) and r.get("success") is False
    r = ASHARE_TOOL_REGISTRY["get_factor_exposure"]("2099-01-01")
    assert isinstance(r, dict) and r.get("success") is False


_MARKET = pathlib.Path("data/ashare_market.duckdb")


@pytest.mark.skipif(not _MARKET.exists(), reason="真实 market 库不存在")
def test_query_universe_on_real_db():
    from ashare.data import query
    query.close_db()                  # 全套联跑时前面的测试可能钉在临时库上，先解钉
    query.open_db(str(_MARKET))       # close 保留旧路径（契约），必须显式指回真库
    r = ASHARE_TOOL_REGISTRY["query_universe"]("2019-06-28", limit=5)
    assert r["success"] is True and r["total"] > 1000
    assert len(r["stocks"]) == 5 and r["truncated"] is True
    assert set(r["stocks"][0]) == {"ts_code", "name", "sw_l1"}


def test_l4_signal_tool_touches_no_write_names():
    """get_signal_list 只许触达 ledger_store 的读函数 —— 写名出现在 agent_tools 源码里即违规。"""
    src = pathlib.Path("ashare/agent_tools.py").read_text(encoding="utf-8")
    for bad in ("save_signal_plan", "record_confirms", "write_positions"):
        assert bad not in src, f"agent_tools 引用了 ledger 写函数 {bad}（D1：LLM 层无写路径）"


def test_signal_list_tool_behaviour(tmp_path, monkeypatch):
    from ashare.data import _ledger, ledger_store
    monkeypatch.setattr(_ledger, "DEFAULT_LEDGER_PATH", str(tmp_path / "led.duckdb"))
    r = ASHARE_TOOL_REGISTRY["get_signal_list"]()
    assert r["success"] is False and "还没有生成过" in r["error"]        # 空库安静
    ledger_store.save_signal_plan({
        "as_of": "2026-08-28", "param_hash": "ph", "data_snapshot_id": "snap",
        "execute_on": "2026-08-31T09:15:00+08:00", "strategy_version": "v1",
        "target_position": 0.8, "position_calibrated": True, "excluded": [],
        "warnings": [], "orders": [{"ts_code": f"{i:06d}.SH", "name": "x", "action": "BUY",
                                    "current_weight": 0, "target_weight": 0.02,
                                    "limit_price_range": [1, 2], "urgency": "normal",
                                    "factor_contrib": {"f": 1}} for i in range(30)]})
    r = ASHARE_TOOL_REGISTRY["get_signal_list"](top=10)
    assert r["success"] is True and r["n_orders"] == 30
    assert len(r["orders"]) == 10 and r["truncated"] is True             # T2 截断
    assert "factor_contrib" not in r["orders"][0]                        # 精简版不带归因
    import json as _json
    assert len(_json.dumps(r, ensure_ascii=False).encode()) < 3072       # T2：< 3KB
    assert "人工执行" in r["note"]


def test_factor_tool_falls_back_and_labels_both_degradations(monkeypatch):
    """两层降级都必须【说出来】，不许静默替换：
    ① 请求日不是周频调仓日 → 回退到不晚于它的最近调仓日（只向过去退，向未来退是前视）；
    ② 因子是在旧数据快照下算的 → 标 stale + 两个快照 id。
    为什么不能只回一句 hint 让模型自己重查：2026-08-28 实测，模型拿到 error 之后
    没有照提示再查，而是直接编了一个答案 —— 比工具报错坏得多。"""
    import datetime as dt
    from ashare.data import query
    from ashare.factors import store as fstore
    import pandas as pd
    req, hit = dt.date(2026, 8, 27), dt.date(2026, 8, 21)
    monkeypatch.setattr(query, "open_db", lambda *a, **k: None)
    monkeypatch.setattr(query, "norm_date", lambda d, **k: req if str(d) == str(req) else d)
    monkeypatch.setattr(query, "get_trade_dates", lambda d, **k: [dt.date(2026, 8, 14), hit])
    monkeypatch.setattr(query, "get_universe", lambda d, **k: ["A.SH", "B.SZ"])
    monkeypatch.setattr(query, "snapshot_id", lambda **k: "NEW")
    monkeypatch.setattr(fstore, "read_current", lambda *a, **k: ({}, []))      # 严格档全落空
    ser = pd.Series([2.0, 1.0], index=["A.SH", "B.SZ"])
    calls: list = []
    def fake_any(h, d, u):
        calls.append(d)
        return ({"reversal_20": (ser, ser)}, "OLD", []) if d == hit else ({}, None, [])
    monkeypatch.setattr(fstore, "read_any_snapshot", fake_any)

    r = ASHARE_TOOL_REGISTRY["get_factor_exposure"]("2026-08-27", factor="reversal_20", top=2)
    assert r["success"] is True
    assert calls[0] == req and calls[1] == hit, f"候选顺序必须由近及远，实际 {calls}"
    assert r["as_of"] == str(hit) and r["requested_as_of"] == str(req)     # ① 说出真正用的日期
    assert r["stale"] is True and r["computed_under_snapshot"] == "OLD"     # ② 说出快照差异
    assert r["current_snapshot"] == "NEW" and "回答时必须说明" in r["note"]
    assert [x["ts_code"] for x in r["top"]] == ["A.SH", "B.SZ"]


def test_relaxed_read_is_whitelisted_to_the_agent_layer():
    """宽松读（无视快照）绝不许进决策路径 —— 回测/信号一律走严格的 read_current。
    白名单钉死在这里：多一个调用方就红。"""
    import subprocess
    out = subprocess.run(["grep", "-rln", "read_any_snapshot\\|read_factor_values_any_snapshot",
                          "--include=*.py", "ashare/"], capture_output=True, text=True).stdout
    callers = {f for f in out.split() if f}
    assert callers == {"ashare/agent_tools.py", "ashare/factors/store.py",
                       "ashare/data/derived_store.py"}, f"宽松读出现了新调用方: {callers}"


def test_fundamentals_tool_is_local_pit_and_within_budget(monkeypatch):
    """个股基本面必须来自本地 PIT 且带公告日 —— 这正是它相对联网源的价值所在
    （联网源给的是"最新财报"，那是前视口径）。返回体守 T2 的 3KB。"""
    import json, pathlib
    if not pathlib.Path("data/ashare_market.duckdb").exists():
        import pytest as _p; _p.skip("真实 market 库不存在")
    from ashare.data import query
    query.close_db(); query.open_db("data/ashare_market.duckdb")
    r = ASHARE_TOOL_REGISTRY["get_stock_fundamentals"]("600519")
    assert r["success"] is True and r["profile"]["name"]
    fin = r["financials"]
    assert fin["announced_on"] and fin["report_period"] and fin["lag_days"] >= 0, \
        "财报必须带公告日/报告期/滞后天数 —— 没有它就分不清 PIT 与前视"
    assert r["price_60d"]["basis"].startswith("后复权")
    assert len(json.dumps(r, ensure_ascii=False).encode()) < 3072
    bad = ASHARE_TOOL_REGISTRY["get_stock_fundamentals"]("AAPL")
    assert bad["success"] is False                      # 非 A 股安静失败，不乱猜


def test_price_levels_are_measured_not_offset_from_current_price():
    """支撑/压力必须是从价格结构量出来的，不能是「当前价 ±固定比例」。
    2026-08-28 用户抓到的正是后者：多只票同一形状、从没跌破过支撑 —— 那是编的。
    两条判据：① 不同票的支撑距离必须显著不同；② 真跌破时必须如实报「无支撑」。"""
    import pathlib, pytest as _p
    if not pathlib.Path("data/ashare_market.duckdb").exists():
        _p.skip("真实 market 库不存在")
    from ashare.data import query
    query.close_db(); query.open_db("data/ashare_market.duckdb")
    g = ASHARE_TOOL_REGISTRY["get_price_levels"]

    gaps = []
    for code in ("600519", "000001", "300750"):
        r = g(code)
        assert r["success"] and r["supports"], code
        for lv in r["supports"] + r["resistances"]:
            assert lv["strength"] >= 1 and lv["confirmed_by"], "每个价位必须带证据链"
        gaps.append(abs(r["supports"][0]["price"] / r["current_price"] - 1))
    assert max(gaps) - min(gaps) > 0.01, \
        f"三只票的支撑距离几乎相同（{gaps}）—— 像是按当前价固定偏移生成的"

    # ② 真实暴跌日必须能报出「下方无支撑」
    none_cnt = 0
    uni = query.get_universe("2015-08-26")[:200]
    p = query.get_price_panel("2015-08-26", uni, "close", lookback=120).ffill()
    rank = ((p.iloc[-1] - p.min()) / (p.max() - p.min())).dropna().sort_values()
    for code in rank.index[:10]:
        r = g(code, as_of="2015-08-26")
        if r.get("success") and not r["supports"]:
            none_cnt += 1
    assert none_cnt > 0, "股灾日一只报「无支撑」的都没有 —— 该分支是死的"
