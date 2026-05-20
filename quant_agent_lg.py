# ============================================================
#  QuantAgent v2.0 — LangGraph 框架版
#  ─────────────────────────────────────────────
#  与 quant_agent.py 共存，通过 USE_LANGGRAPH 环境变量切换
#
#  架构：StateGraph 编排
#  ┌─────────────────────────────────────────────────────┐
#  │  llm        调用 DeepSeek，决定下一步                 │
#  │   ↓ has_tool_calls?                                  │
#  │  ├─ yes → tools      并行执行所有工具调用            │
#  │  │         └─→ 回到 llm                              │
#  │  └─ no  → render     服务端 Markdown → HTML           │
#  │           └─→ followup  生成 3 个建议问题             │
#  │                └─→ END                               │
#  └─────────────────────────────────────────────────────┘
#
#  与原版 stream_quant_agent 的差异：
#  1. 用 LangGraph StateGraph 编排（取代 while 循环）
#  2. tools_node 自动并行（节省时间）
#  3. 节点级 retry / checkpoint 能力（待用）
#  4. 同样的 SSE 事件协议，前端无感
# ============================================================

import json
import re
import time
import datetime
from typing import TypedDict, Annotated, Sequence
from concurrent.futures import ThreadPoolExecutor

# Token 批处理参数（环境变量可覆盖）
import os as _os
_TOKEN_FLUSH_MS = int(_os.getenv("LG_TOKEN_FLUSH_MS", "30"))        # 30ms 批量窗口
_TOKEN_FLUSH_CHARS = int(_os.getenv("LG_TOKEN_FLUSH_CHARS", "40"))  # 或累计 40 字符

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage,
)

# 复用原 quant_agent 里的工具实现 + LLM 客户端配置
from quant_agent import (
    TOOL_REGISTRY, TOOL_SCHEMAS, SYSTEM_PROMPT,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    render_markdown_to_html, _generate_followups, dispatch_tool,
    _sanitize_messages,
    TRADING_TOOLS, _is_request_authenticated,
    _set_request_device_id, _set_request_authenticated,
    _get_request_device_id,
    _build_system_prompt,
)

# ════════════════════════════════════════════════════════════
# 1. LLM 客户端（LangChain 风格）
# ════════════════════════════════════════════════════════════

_llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY or "not-set",
    base_url="https://api.deepseek.com",
    max_tokens=8000,
    streaming=True,
)


# ════════════════════════════════════════════════════════════
# 2. 工具适配：把现有 TOOL_REGISTRY 包装为 LangChain Tool
# ════════════════════════════════════════════════════════════

def _make_lc_tool(name: str):
    """
    把 quant_agent.TOOL_REGISTRY 里的函数包成 StructuredTool。
    保留原 schema 的输入约束（通过 OpenAI 函数式协议）。
    """
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == name)

    def _impl(**kwargs):
        return dispatch_tool(name, kwargs)

    return StructuredTool.from_function(
        func=_impl,
        name=name,
        description=schema["description"],
        # 让 LangChain 用原 OpenAI 风格的参数 schema
        args_schema=None,
        infer_schema=False,
    )


_LC_TOOLS = [_make_lc_tool(n) for n in TOOL_REGISTRY.keys()]

# 把 OpenAI 格式工具直接绑定到 LLM（LangChain 会自动识别）
def _to_openai_tools(schemas):
    return [
        {"type": "function",
         "function": {"name": s["name"],
                       "description": s["description"],
                       "parameters": s["input_schema"]}}
        for s in schemas
    ]

_llm_with_tools = _llm.bind_tools(_to_openai_tools(TOOL_SCHEMAS))
# 未登录用户专用：去掉交易类工具
_llm_with_tools_anon = _llm.bind_tools(
    _to_openai_tools([s for s in TOOL_SCHEMAS
                       if s["name"] not in TRADING_TOOLS])
)


def _llm_for_request():
    """按当前请求鉴权状态选择 LLM 实例"""
    return _llm_with_tools if _is_request_authenticated() else _llm_with_tools_anon


# ════════════════════════════════════════════════════════════
# 3. State 定义
# ════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    final_text: str
    final_html: str
    suggestions: list
    iteration: int


# ════════════════════════════════════════════════════════════
# 4. 节点实现
# ════════════════════════════════════════════════════════════

import asyncio as _asyncio


async def _llm_node(state: AgentState, config=None) -> dict:
    """
    异步流式调用 LLM。
    每轮开始前先 dispatch round_start，让前端清空上一轮的流式残留
    （上一轮如果说了"我来查行情..."这种过渡话，会被最终答案覆盖）。
    """
    # 新一轮 → 通知前端清空已累积的 streaming 缓冲
    await adispatch_custom_event("round_start", {}, config=config)

    full = None
    # ── Token 批处理：30ms 或 40 字符触发一次 dispatch ──
    # 避免每 token 一次 callback 链路的开销（实测可省 1.5–2.5s）
    buf = []
    buf_len = 0
    last_flush = time.monotonic()

    async def _flush():
        nonlocal buf, buf_len, last_flush
        if buf:
            await adispatch_custom_event(
                "llm_token", {"text": "".join(buf)}, config=config)
            buf = []
            buf_len = 0
            last_flush = time.monotonic()

    async for chunk in _llm_for_request().astream(state["messages"], config=config):
        full = chunk if full is None else full + chunk
        content = getattr(chunk, "content", "") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content
                                if isinstance(p, dict))
        if content:
            buf.append(content)
            buf_len += len(content)
            elapsed_ms = (time.monotonic() - last_flush) * 1000
            if elapsed_ms >= _TOKEN_FLUSH_MS or buf_len >= _TOKEN_FLUSH_CHARS:
                await _flush()

    # 尾部残余必须发出
    await _flush()

    return {
        "messages": [full] if full else [],
        "iteration": state.get("iteration", 0) + 1,
    }


async def _tools_node(state: AgentState, config=None) -> dict:
    """
    并行执行所有 tool_calls。
    每个工具的 start/end 通过 dispatch_custom_event 推到外层 SSE。
    """
    last = state["messages"][-1]
    if not hasattr(last, 'tool_calls') or not last.tool_calls:
        return {"messages": []}

    def _extract(tc):
        name = tc.get("name") if isinstance(tc, dict) else tc.name
        args = tc.get("args") if isinstance(tc, dict) else tc.args
        tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
        return name, args or {}, tc_id

    # 先全部 yield tool_call 事件（用户立刻看到状态）
    for tc in last.tool_calls:
        name, args, tc_id = _extract(tc)
        await adispatch_custom_event(
            "tool_call_start",
            {"name": name, "input": args, "id": tc_id},
            config=config,
        )

    # 并行执行
    loop = _asyncio.get_event_loop()
    # 在主线程读出当前 auth/device 状态，传给 worker 线程重新注入
    # （否则 run_in_executor 的 worker 看不到主线程的 thread-local）
    _ctx_authed = _is_request_authenticated()
    _ctx_device = _get_request_device_id()

    def _run_one(tc):
        # worker 线程：必须重新设置 thread-local，否则 dispatch_tool 守门会误判
        _set_request_device_id(_ctx_device)
        _set_request_authenticated(_ctx_authed)
        name, args, tc_id = _extract(tc)
        try:
            result = dispatch_tool(name, args)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        content = json.dumps(result, ensure_ascii=False, default=str)
        if len(content) > 3000:
            if isinstance(result, dict):
                for k in ("results", "news", "quotes"):
                    if isinstance(result.get(k), list) and len(result[k]) > 3:
                        result[k] = result[k][:3]
                        result["truncated"] = True
            content = json.dumps(result, ensure_ascii=False, default=str)
        return name, tc_id, result, ToolMessage(
            content=content, tool_call_id=tc_id, name=name)

    coros = [loop.run_in_executor(None, _run_one, tc) for tc in last.tool_calls]
    completed = await _asyncio.gather(*coros)

    # 每个工具完成后推 tool_result 事件
    tool_msgs = []
    for name, tc_id, result, msg in completed:
        is_err = isinstance(result, dict) and not result.get("success", True)
        await adispatch_custom_event(
            "tool_call_end",
            {"name": name, "id": tc_id, "result": result, "is_error": is_err},
            config=config,
        )
        tool_msgs.append(msg)

    return {"messages": tool_msgs}


async def _render_node(state: AgentState) -> dict:
    """从最后一条 AIMessage 中提取文本，服务端渲染 HTML"""
    final_text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.content:
            content = m.content
            if isinstance(content, list):
                final_text = "".join(p.get("text", "") for p in content
                                       if isinstance(p, dict))
            else:
                final_text = str(content)
            break
    return {
        "final_text": final_text,
        "final_html": render_markdown_to_html(final_text),
    }


async def _followup_node(state: AgentState) -> dict:
    """生成 3 个 follow-up 问题（复用原实现，跑在线程里避免阻塞 event loop）"""
    msgs_dict = []
    for m in state["messages"]:
        if isinstance(m, SystemMessage):
            msgs_dict.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            msgs_dict.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msgs_dict.append({"role": "assistant", "content": m.content or ""})
    loop = _asyncio.get_event_loop()
    items = await loop.run_in_executor(None, _generate_followups, msgs_dict)
    return {"suggestions": items}


# ════════════════════════════════════════════════════════════
# 5. 条件路由
# ════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════
# 6. 构建 Graph
# ════════════════════════════════════════════════════════════

def _should_finalize(state: AgentState) -> list:
    """
    LLM 输出无 tool_calls 时，并行触发 render + followup（互不依赖）。
    LangGraph 看到 list 返回会同时执行两个节点。
    """
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and getattr(last, 'tool_calls', None):
        return ["tools"]
    return ["render", "followup"]   # ⭐ 并行


def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("llm",       _llm_node)
    builder.add_node("tools",     _tools_node)
    builder.add_node("render",    _render_node)
    builder.add_node("followup",  _followup_node)

    builder.set_entry_point("llm")
    # 关键：用 list 返回让 LangGraph 并行调度 render + followup
    builder.add_conditional_edges("llm", _should_finalize,
                                   ["tools", "render", "followup"])
    builder.add_edge("tools", "llm")
    builder.add_edge("render", END)
    builder.add_edge("followup", END)
    return builder.compile()


_graph = _build_graph()


# ════════════════════════════════════════════════════════════
# 7. SSE 适配层：与 quant_agent.stream_quant_agent 同协议
# ════════════════════════════════════════════════════════════

def _openai_to_lc_messages(openai_msgs: list) -> list:
    """server.py 传进来的 OpenAI 风格消息 → LangChain 消息对象"""
    out = []
    for m in openai_msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls:
                # 反向构造 AIMessage with tool_calls
                lc_tcs = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    lc_tcs.append({
                        "name": fn.get("name"),
                        "args": args,
                        "id": tc.get("id"),
                    })
                out.append(AIMessage(content=content or "", tool_calls=lc_tcs))
            else:
                out.append(AIMessage(content=content or ""))
        elif role == "tool":
            out.append(ToolMessage(content=content,
                                     tool_call_id=m.get("tool_call_id"),
                                     name=m.get("name", "")))
    return out


def _lc_to_openai_messages(lc_msgs: list) -> list:
    """LC 消息 → OpenAI 风格（写回 session.messages 用）"""
    out = []
    for m in lc_msgs:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            d = {"role": "assistant", "content": m.content or ""}
            if getattr(m, 'tool_calls', None):
                d["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                   "arguments": json.dumps(tc.get("args", {}),
                                                            ensure_ascii=False)}}
                    for tc in m.tool_calls
                ]
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool",
                         "tool_call_id": m.tool_call_id,
                         "content": m.content})
    return out


async def stream_quant_agent_lg(messages: list, max_iterations: int = 15):
    """
    LangGraph 版异步生成器主循环 —— 支持 token 级流式。
    用 graph.astream_events(version="v2") 拿到细粒度事件并映射到原 SSE 协议。

    yield 事件类型（与原版完全一致）：
      - content_delta:  每个 LLM token（NEW）
      - tool_call:      工具调用开始
      - tool_result:    工具调用结果
      - final:          最终答案 + HTML
      - suggestions:    follow-up 建议
      - error:          异常
    """
    _sanitize_messages(messages)
    # 动态拼装 system prompt（含当前用户档案 + Top-3 情景记忆）
    sys_prompt = _build_system_prompt(messages)
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": sys_prompt})
    else:
        messages[0]["content"] = sys_prompt

    lc_msgs = _openai_to_lc_messages(messages)
    state = {"messages": lc_msgs, "iteration": 0}
    config = {"recursion_limit": max_iterations * 4}

    final_state_output = None
    seen_tool_runs = set()    # 避免 LangChain 偶发重放重复 tool 事件
    iteration = 0

    try:
        async for event in _graph.astream_events(state, config=config,
                                                  version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {}) or {}

            # ── 1. 自定义事件：round / token / 工具调用（从节点 dispatch）──
            if kind == "on_custom_event":
                if name == "round_start":
                    # 新一轮 LLM 输出开始 → 前端清空 streaming 缓冲
                    yield {"type": "content_reset"}
                elif name == "llm_token":
                    text = (data or {}).get("text", "")
                    if text:
                        yield {
                            "type": "content_delta",
                            "text": text,
                            "iteration": iteration,
                        }
                elif name == "tool_call_start":
                    d = data or {}
                    tc_id = d.get("id", "")
                    if tc_id and tc_id not in seen_tool_runs:
                        seen_tool_runs.add(tc_id)
                        yield {
                            "type": "tool_call",
                            "name": d.get("name", ""),
                            "input": d.get("input", {}),
                            "id": tc_id,
                        }
                elif name == "tool_call_end":
                    d = data or {}
                    result = d.get("result", {})
                    yield {
                        "type": "tool_result",
                        "name": d.get("name", ""),
                        "result": result if isinstance(result, dict)
                                  else {"value": result},
                        "is_error": d.get("is_error", False),
                    }

            # ── 2. LLM 调用结束（追踪迭代轮次）──
            elif kind == "on_chat_model_end":
                iteration += 1

            # ── 3. 节点结束（提取 render/followup/全图的输出）──
            elif kind == "on_chain_end":
                metadata = event.get("metadata", {}) or {}
                node = metadata.get("langgraph_node", "") or name
                output = data.get("output", {})

                if node == "render":
                    if isinstance(output, dict):
                        yield {
                            "type": "final",
                            "text": output.get("final_text", ""),
                            "text_html": output.get("final_html", ""),
                            "iterations": iteration,
                        }

                elif node == "followup":
                    items = (output.get("suggestions")
                             if isinstance(output, dict) else None) or []
                    if items:
                        yield {"type": "suggestions", "items": items}

                elif name == "LangGraph":
                    # 全图结束，保留最终 state 用于消息回写
                    if isinstance(output, dict):
                        final_state_output = output

        # 写回 messages 让 server.py 持久化用
        if final_state_output and final_state_output.get("messages"):
            new_openai = _lc_to_openai_messages(list(final_state_output["messages"]))
            messages.clear()
            messages.extend(new_openai)

    except Exception as e:
        yield {"type": "error", "error": f"LangGraph 执行异常: {e}"}


# ════════════════════════════════════════════════════════════
# 8. CLI 测试入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("LangGraph 版主循环加载成功 ✅")
    print(f"  - Tools 注册数: {len(_LC_TOOLS)}")
    print(f"  - Graph 节点: llm / tools / render / followup")
    print(f"  - Model: {DEEPSEEK_MODEL}")
