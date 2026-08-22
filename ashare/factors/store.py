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
