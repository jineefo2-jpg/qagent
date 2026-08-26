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
    r = ASHARE_TOOL_REGISTRY["query_universe"]("2019-06-28", limit=5)
    assert r["success"] is True and r["total"] > 1000
    assert len(r["stocks"]) == 5 and r["truncated"] is True
    assert set(r["stocks"][0]) == {"ts_code", "name", "sw_l1"}
