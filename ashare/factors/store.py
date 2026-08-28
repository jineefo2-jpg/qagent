"""因子预计算落库 —— 「算 → 写」的编排层，是**唯一**写 `factor_value` 的因子模块。

★ 为什么这一半在 `ashare/factors/` 而写库那一半在 `ashare/data/derived_store.py`
  （架构 §4.3 + 2026-08-21 修正）：`build` 必须调 `compute_panel`，而 `derived_store`
  绝不能反向 import `ashare.factors`；同时分层闸 L1 只许 `ashare/data/**` `import duckdb`。
  两条约束把这件事切成两片：本文件**不 import duckdb**（L1 通过），落库全部经
  `derived_store` 的 DataFrame 接口。这不是「包一层转发」—— 白名单、幂等判定、
  raw/processed 两遍计算都是真逻辑。

★ 因子一律经 `compute_factor` / `compute_panel` 取，不直接调 `get_factor(n).fn(...)`：
  直连会绕过四道保护 —— universe 校验（18 个因子的唯一检查点）、让因子的部分返回变安全的
  `.reindex(codes)`、`spec.default_params`（落库的 `param_hash` 哈希的正是它，分家就等于
  库里存着一份「参数写着 A、内容是 B」的因子值，且是主键）、以及 `available_from` 短路。
"""
from __future__ import annotations
import datetime as _dt
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ashare.data import derived_store, query
from .base import ALPHA_CATEGORIES, compute_panel, get_factor


def build(names: Sequence[str], dates: Sequence[_dt.date], *,
          overwrite: bool = False,
          progress: Optional[Callable[[int, int], None]] = None
          ) -> Tuple[Dict[str, int], List[str]]:
    """预计算 `names × dates` 并写入 `factor_value`，返回 `({因子: 写入行数}, warnings)`。

    每个 (因子, 日期) 落 `len(get_universe(日期))` 行，`raw_value` / `processed_value` 两列都写。

    ★ 返回值带 warnings（global-constraints ★，同一缺陷此前已出现两次）：中性化被跳过、
      因子未到 `available_from` 都会照常落库，而落下的行**没有任何列**记着那一天是降级
      算出来的。吞掉 warning，后面从库里读因子跑出来的净值曲线看起来完全正常。

    ★ 幂等判据是 `snapshot_id`，不是「主键下有行」：`param_hash` 哈希
      `name + default_params + neutralize + available_from`（2026-08-22 评审 I5 把后两个
      折了进去 —— 它们都改变落库的值），但 **因子函数体不在里面**（也没法哈希）。
      所以同一个主键下仍然可能存着另一种语义的值 —— `overwrite=False` 只保证
      「这一代是当前这批数据算的」，**不保证它与当前实现一致**。
      改了函数体想重算，必须显式 `overwrite=True`，库里不会自己发现。

    ★ `snapshot_id(pin=True)`：全程一个快照写进所有行。不钉住的话，跑到一半撞上 promote
      会静默重连，于是半数行是另一个数据库算的、却统一盖着开跑时那个指纹（D7 失效）。
      ⚠ **钉子的生命周期长过本函数**：正常返回后 `query` 仍然钉着这个 inode，
      只有 `query.close_db()` 解得开。长驻进程（server.py）里跑完 build 不 close，
      下一次 nightly promote 之后每一个 A 股工具都会抛「请重跑」。
      `open_db()` **不是解药**：它只把连接指到新文件上，那一次检查于是静默通过 ——
      钉子还举着，而你已经在读另一个数据库了（再 promote 一次照样抛）。
      **解钉是调用方的事**：批处理脚本跑完 `close_db()`，测试放进 fixture。
    """
    cols = list(names)
    if not cols:
        # 与 compute_panel 同口径（「没有因子的面板没有意义」）。静默返回 {} 会让
        # 「名字列表算错成空」的调用方看到一次成功的空跑，还顺手钉住了快照。
        raise ValueError("names 为空：没有因子要预计算")
    specs = {n: get_factor(n) for n in cols}            # 名字拼错在这里 KeyError，不是空结果

    # ★ alpha 白名单，且**在算任何一个因子之前**全部验完 —— 边算边验会让前半张表已经落库。
    bad = {n: s.category for n, s in specs.items() if s.category not in ALPHA_CATEGORIES}
    if bad:
        raise ValueError(
            f"因子 {sorted(bad)} 不在 alpha 白名单 {ALPHA_CATEGORIES}，不能落库（实际 category：{bad}）。\n"
            f"  · industry 返回的是 category dtype 的字符串，而 raw_value 是 DOUBLE —— "
            f"遍历注册表落库要么在它这里抛，要么更糟：静默写进一列 NULL，"
            f"看起来「行业因子存在且全空」。\n"
            f"  · log_mv / beta_250 是数值，存得下，但它们是中性化的回归元不是 alpha；"
            f"拿来当信号在 A 股历史回测里【非常好看】。要存由调用方显式指定，"
            f"不能靠遍历 FACTOR_REGISTRY 撞上。")

    hashes = {n: s.param_hash() for n, s in specs.items()}
    # 去重：`done` 在循环外算一次，重复日期会走两遍 todo 分支，把同一批行写两遍
    # （UPSERT 幂等，所以库是对的）而 written 报双倍 —— 一份对不上库的进度数字。
    ds = list(dict.fromkeys(query.norm_date(d, name="trade_date") for d in dates))
    written = {n: 0 for n in cols}
    warns: List[str] = []

    snap = query.snapshot_id(pin=True)
    done = set() if overwrite else derived_store.current_factor_dates(hashes, ds)

    for i, d in enumerate(ds, 1):
        todo = [n for n in cols if (n, d) not in done]
        if todo:
            codes = query.get_universe(d)               # D5：股票池按 as_of 动态生成
            raw, _ = compute_panel(todo, d, codes, processed=False)
            # ponytail: 每个因子算两遍（raw 一遍、processed 一遍），换取只走 compute_panel
            # 一个入口。想省这一半，就给 base 加一个同时返回两者的入口 —— 别在这里手工拼
            # pipeline.process：那会绕过 available_from 短路，把全 NaN 的一天 fillna 成 0。
            proc, w = compute_panel(todo, d, codes, processed=True)
            # 只收 processed 那遍的 warning：raw 那遍的是它的子集（processed=False 时
            # compute_factor 直接 `return raw, []`，只有 available_from 短路会出声）。
            warns += w
            derived_store.write_factor_values(_long_frame(todo, hashes, d, codes, raw, proc, snap))
            # 池子缩了就把旧成员的行删掉：留着 `current_factor_dates` 的 bool_and 对这一天
            # 永远为假 —— 每次跑都重算，overwrite=False 的跳过永久失效（评审 I1）。
            gone = derived_store.drop_out_of_universe({n: hashes[n] for n in todo}, d, codes)
            if gone:
                warns.append(f"{d} 删除 {gone} 行不在当前股票池里的旧因子值："
                             f"该日的池子随数据修正变了（这些行 read 本来就读不到）")
            for n in todo:
                written[n] += len(codes)
        if progress is not None:
            progress(i, len(ds))
    return written, warns


_READ_MEMO: dict = {}           # 单条：{"key": ..., "val": ...}，见 read_current 内的注释
_WINDOW: dict = {}              # 整窗预取，见 preload_window


def preload_window(param_hashes: Mapping[str, str], dates: Sequence) -> None:
    """回测入口的一次性预取：把 read_current 的「逐日开连接 + 三次查询」折成
    1 次批量判命中 + 1 次整窗读 + 逐日 pivot。键含 snapshot_id —— 换库自动失效；
    read_current 对窗口外的日期、或没预取过的进程，照走逐日路径。契约不变：
    判命中仍是对当前快照判，池子核对与整批作废的判词都在 read_current 原位。"""
    if not param_hashes:
        return
    ds = sorted({query.norm_date(x, name="date") for x in dates})
    if not ds:
        return
    snap = query.snapshot_id()
    cur = derived_store.current_factor_dates(param_hashes, ds)
    long = derived_store.read_factor_window(param_hashes, ds[0], ds[-1])
    names = list(param_hashes)
    by_date: dict = {}
    if len(long):
        for d, g in long.groupby("trade_date"):
            raw = (g.pivot(index="ts_code", columns="factor_name", values="raw_value")
                    .reindex(columns=names).rename_axis(columns=None))
            proc = (g.pivot(index="ts_code", columns="factor_name", values="processed_value")
                     .reindex(columns=names).rename_axis(columns=None))
            by_date[d] = (raw, proc)
    _WINDOW["key"] = (tuple(sorted(param_hashes.items())), snap)
    _WINDOW["dates"] = set(ds)
    _WINDOW["cur"] = cur
    _WINDOW["by_date"] = by_date


def read_current(param_hashes: Mapping[str, str], as_of_date, universe: Sequence[str]
                 ) -> Tuple[Dict[str, Tuple[pd.Series, pd.Series]], List[str]]:
    """读回**可以直接当现算用**的因子值：`({因子: (raw, processed)}, warnings)`。

    `combine(use_store=True)` 的唯一取数口。没命中的因子不出现在返回的 dict 里
    （由调用方现算），本函数**绝不现算** —— 现算与落库的口径分歧是最难查的一类 bug
    （`derived_store.read_factor_values` 的同一条裁决）。

    ★ **一次命中到底保证了什么，一个字都不能多**：库里这一天的**每一行**都盖着
      当前的 `snapshot_id`（`current_factor_dates` 的 `bool_and`），且行取自
      `(factor_name, param_hash)` 这个主键。而 `param_hash` 只覆盖
      `default_params` + `neutralize` + `available_from`（`base._HASHED_SPEC_FIELDS`），
      **不覆盖因子函数体**（也没法覆盖）。所以命中只等于
      「某一代实现、在同一组声明参数、同一批数据上算出来的值」——
      这是现有主键能给的最强保证，不是「与当前实现一致」。
      改了函数体想让缓存跟上，只能 `build(..., overwrite=True)` 显式说出来，
      库自己发现不了（与 `build` 的幂等判据是同一条契约，见那边的 ★）。

    ★ 判命中用 `current_factor_dates` 而不是「`read` 返回的行数」：`read` 自己那句
      `snapshot_id = ?` 只把陈旧的**行**滤掉，于是一天里坏一行 = 那只票读回 NaN，
      覆盖率闸多半照样放行 —— 一个缺了几只票的横截面，与「这几只票今天没值」
      逐位相同。整天都当前才算命中。

    ★ 池子必须与 `get_universe(d)` 逐只相同，否则整批不用：`processed_value` 是
      `pipeline.process` 的产物，而去极值（中位数）、中性化（横截面 OLS）、zscore
      三步**全是横截面统计量** —— 同一只票在 3,100 只的池子里和在 10 只的池子里
      算出来是两个数，且两个都是合理的浮点。`build` 落的是 `get_universe(d)` 那个
      横截面（`drop_out_of_universe` 保证库里不多不少就是它），所以这条等式成立时
      「读回来的 == 现算的」才是真话。
      顺序上它排在命中判定【之后】：冷库（第一次跑）根本不该为此多打一次
      `explain_universe` 的全表扫描。

    ponytail: 逐日三次查询（判命中 / 取 raw / 取 processed），**天花板卡在这里**。
      2026-08-24 在架构 §1.1 的真实规模上实测（16 因子 × 3,100 只 × 780 周
      = 3,869 万行 / 2.2 GB 派生库）：本函数 49 ms/日 → 38 s/次回测，而它省下的
      `pipeline.process` 是 45 ms/日 → 35 s/次。**净收益约等于零**（相对现算 1.05×）。
      钱花在哪：`derived_store._read_conn()` 每次调用现开一个 DuckDB 只读连接
      （2.2 GB 库上 5.8 ms × 3 = 17 ms），加上 `read_factor_values` 每天两次
      `pivot + reindex`（约 20 ms）；真正的扫描只有 4~5 ms。
      同一份数据、连接常开、一次取两列、不 pivot：**4.3 ms/日 → 3.4 s/次**，
      也就是 §1.2 那行「净值-only < 8 s」够得着的唯一形状。
      升级路径不在本文件（L1 挡着，`store.py` 不能碰 duckdb）：要给
      `derived_store` 加一个**批量**口（一个连接、一次查一整段日期、二维 ndarray 出），
      再由 `engine` 在回测入口预取一次。那是另一张变更单。
      **在此之前 `use_store=True` 的收益全部来自「不跑因子函数」那一半**
      （滚动窗口 / PIT 关联 / 取数 —— §1.2 给全量预计算的预算是 20 min，
      折算约 770 ms/日/16 因子，本函数是它的 1/15），不是来自省中性化。
    """
    if not param_hashes:
        return {}, []
    d = query.norm_date(as_of_date, name="as_of_date")
    codes = list(universe)

    # 单条备忘录：引擎一个调仓日 combine 与诊断 compute_panel 背靠背查同参（2026-08-27
    # 性能 profile：本函数占 44%，其中一半是这对重复）。只存最近一条。
    # ★ 键必须含 query.snapshot_id()（约 1ms，远小于省下的两次 80ms 读）：
    #   「判命中」的语义是**对当前数据快照**判 —— promote 换库后备忘录若还回放旧命中，
    #   等于把 test_a_snapshot_change_sends_the_store_path_back_to_computing 钉死的
    #   契约整个绕过。快照一换，键就变，判命中照常重跑。
    snap = query.snapshot_id()
    memo_key = (d, tuple(sorted(param_hashes.items())), hash(tuple(codes)), snap)
    if _READ_MEMO.get("key") == memo_key:
        got, warns = _READ_MEMO["val"]
        return {n: (r.copy(), p.copy()) for n, (r, p) in got.items()}, list(warns)

    # 整窗预取命中（preload_window）→ 判命中用批量结果、取值用内存 pivot，零 SQL。
    # 窗口外的日期 / 没预取的进程 / 快照已换 → win=False，照走下面的逐日路径。
    win = _WINDOW.get("key") == (memo_key[1], snap) and d in _WINDOW["dates"]
    cur = _WINDOW["cur"] if win else derived_store.current_factor_dates(param_hashes, [d])
    hit = {n: h for n, h in param_hashes.items() if (n, d) in cur}
    if not hit:
        return {}, []

    pool = query.get_universe(d)
    if set(codes) != set(pool):
        return {}, [f"{d} 因子缓存整批跳过：传入的股票池（{len(codes)} 只）不是当日 "
                    f"get_universe 的池子（{len(pool)} 只），而落库的 processed_value 是在"
                    f"建库那个横截面上算的（去极值/中性化/zscore 全是横截面统计量），"
                    f"换个池子它就不是同一个数 —— 改为现算"]

    if win:
        pair = _WINDOW["by_date"].get(d)
        if pair is None:
            # 批量判说在、窗口里却没有行：preload 的两次查询之间库被改了。同逐日路径的
            # 作废语义 —— 整批作废现算，且必须出声。
            return {}, [f"{d} 因子缓存整批作废：窗口判命中与取值不一致，改为现算"]
        raw_w, proc_w = pair
        out = {n: (raw_w[n].reindex(codes), proc_w[n].reindex(codes)) for n in hit}
        _READ_MEMO["key"], _READ_MEMO["val"] = memo_key, (out, [])
        return {n: (r.copy(), p.copy()) for n, (r, p) in out.items()}, []

    raw, w1 = derived_store.read_factor_values(hit, d, codes, processed=False)
    proc, w2 = derived_store.read_factor_values(hit, d, codes, processed=True)
    if w1 or w2:
        # current_factor_dates 说在、read 说不在：两次查询之间换了库（未钉住的调用方）。
        # 此时两列各自命中的是哪一代已经说不清，整批作废现算 —— 而且必须出声。
        return {}, [f"{d} 因子缓存整批作废：判命中与取值之间数据库变了，改为现算"] + w1 + w2
    out = {n: (raw[n], proc[n]) for n in hit}
    _READ_MEMO["key"], _READ_MEMO["val"] = memo_key, (out, [])
    return {n: (r.copy(), p.copy()) for n, (r, p) in out.items()}, []


def _long_frame(todo: Sequence[str], hashes: Mapping[str, str], d: _dt.date,
                codes: Sequence[str], raw: pd.DataFrame, proc: pd.DataFrame,
                snap: str) -> pd.DataFrame:
    """(因子 × 股票) 的宽表摊成 factor_value 的长表。因子外层、股票内层，两列顺序一致。"""
    return pd.DataFrame({
        "factor_name": [n for n in todo for _ in codes],
        "param_hash": [hashes[n] for n in todo for _ in codes],
        "trade_date": d,
        "ts_code": list(codes) * len(todo),
        "raw_value": pd.concat([raw[n] for n in todo], ignore_index=True),
        "processed_value": pd.concat([proc[n] for n in todo], ignore_index=True),
        "snapshot_id": snap,
    })


def read_any_snapshot(param_hashes: Mapping[str, str], as_of_date, universe: Sequence[str]
                      ) -> "Tuple[Dict[str, pd.Series], Optional[str], List[str]]":
    """**只给 agent 的只读探索用**：不管快照是否当前，取最近算出的 processed 值。
    返回 `({因子: processed}, 用的 snapshot_id, warnings)`。

    与 `read_current` 的分工是硬的：回测与信号生成一律走 `read_current`（严格判当前
    快照，D7 要的就是「这次结果由哪批数据算出」）。本函数存在只因为 `snapshot_id`
    是全库指纹、分不清「尾部追加新交易日」与「历史被修正」—— 每日增量做的是前者，
    历史因子值数值上仍然正确，却会被集体判陈旧，让 agent 的历史查询天天罢工。
    调用方必须把 snapshot_id 显示给人看（`ashare/agent_tools.py` 就是这么做的），
    调用方白名单由 tests 钉死。"""
    # 形状与 read_current 严格一致（{因子: (raw, processed)}）—— 调用方在两条路径之间
    # 切换时不该还要记得形状不同；不一致过一次，下游 `for n, (_, proc) in ...` 当场炸。
    proc, snap, warns = derived_store.read_factor_values_any_snapshot(
        param_hashes, as_of_date, universe, processed=True)
    raw, _, _ = derived_store.read_factor_values_any_snapshot(
        param_hashes, as_of_date, universe, processed=False)
    if proc.empty:
        return {}, None, warns
    return ({n: (raw[n] if n in raw.columns else proc[n] * float("nan"), proc[n])
             for n in proc.columns if proc[n].notna().any()}, snap, warns)
