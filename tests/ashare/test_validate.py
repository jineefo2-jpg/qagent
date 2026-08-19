"""Task 13：validate.py 六项落地校验 + BaoStock 双源交叉（规格 §4.4）。只读校验，任一阻断项失败 → ValidationError。"""
from __future__ import annotations
import datetime as dt
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
from ashare.data import _db, validate
from ashare.data.validate import CheckResult, ValidationError

D = dt.date


def _mutate(path, sql, params=()):
    w = _db.connect_write(path)
    w.execute(sql, list(params))
    w.close()


def _by_name(results):
    return {r.name: r for r in results}


# ══════════════ 干净库：全过 ══════════════
def test_clean_fixture_passes_all_blocking_checks(market_db):
    res = validate.run_all(market_db)
    m = _by_name(res)
    assert set(m) == {"row_completeness", "placeholder_rows", "adj_factor_jumps", "financial_ann_date",
                      "macro_publish_date", "limit_coverage", "frozen_days", "industry_source",
                      "zero_volume_rows", "cross_source"}          # 相等而非包含：删掉任一检查都要失败
    assert all(r.passed for r in res if r.blocking and not r.skipped)
    assert m["cross_source"].skipped is True            # 未提供 BaoStock → SKIPPED，不是 PASS


# ══════════════ 行数完整性（误差为 0）══════════════
def test_missing_daily_bar_row_fails_with_location(market_db):
    _mutate(market_db, "DELETE FROM daily_bar WHERE ts_code='A00001.SZ' AND trade_date=DATE '2024-01-10'")
    r = validate.check_row_completeness(market_db)
    assert r.passed is False and r.blocking is True
    assert any(x["ts_code"] == "A00001.SZ" and x["missing"] == 1 for x in r.detail["stocks"])
    with pytest.raises(ValidationError):
        validate.run_all(market_db)


def test_extra_row_outside_listing_window_fails(market_db):
    """幸存者偏差的反面：未上市区间凭空出现行情也是错。"""
    _mutate(market_db, "INSERT INTO daily_bar (ts_code, trade_date, close, is_suspended) VALUES ('D00004.SZ', DATE '2023-11-30', 1.0, FALSE)")
    r = validate.check_row_completeness(market_db)
    assert r.passed is False


# ══════════════ 占位行合规 ══════════════
def test_placeholder_row_with_volume_fails(market_db):
    _mutate(market_db, "UPDATE daily_bar SET vol = 5 WHERE ts_code='A00001.SZ' AND trade_date=DATE '2024-01-16'")
    r = validate.check_placeholder_rows(market_db)
    assert r.passed is False and r.blocking is True
    assert r.detail["bad_rows"] == 1


def test_placeholder_row_with_price_move_fails(market_db):
    _mutate(market_db, "UPDATE daily_bar SET high = close + 1 WHERE ts_code='A00001.SZ' AND trade_date=DATE '2024-01-16'")
    assert validate.check_placeholder_rows(market_db).passed is False


# ══════════════ 复权因子跳变（告警级）══════════════
def test_adj_factor_jump_is_warning_not_blocking(market_db):
    _mutate(market_db, "UPDATE daily_bar SET adj_factor = 3.0 WHERE ts_code='B00002.SZ' AND trade_date >= DATE '2024-01-10'")
    r = validate.check_adj_factor_jumps(market_db)
    assert r.passed is False and r.blocking is False
    assert r.detail["jumps"][0]["ts_code"] == "B00002.SZ" and r.detail["jumps"][0]["trade_date"] == D(2024,1,10)
    validate.run_all(market_db)                          # 不抛


# ══════════════ 财报 ann_date / 宏观 publish_date ══════════════
def test_financial_ann_date_check(market_db):
    r = validate.check_financial_ann_date(market_db)
    assert r.passed and r.blocking
    # ann_date 晚于入库时间是可疑的（未来公告日）
    _mutate(market_db, "INSERT INTO financial_pit (ts_code, ann_date, end_date, report_type, update_flag) "
                       "VALUES ('A00001.SZ', DATE '2099-01-01', DATE '2098-12-31', '1', 0)")
    r2 = validate.check_financial_ann_date(market_db)
    assert r2.passed is False and r2.detail["future_ann_date"] == 1


def test_macro_publish_date_check(market_db):
    _mutate(market_db, "INSERT INTO macro_indicator VALUES ('m2_yoy', DATE '2024-01-31', DATE '2024-01-15', 8.7, 'rule', current_timestamp)")
    r = validate.check_macro_publish_date(market_db)
    assert r.passed is False and r.blocking is True
    assert r.detail["publish_before_period"] == 1


# ══════════════ 涨跌停覆盖率（报告级）══════════════
def test_limit_coverage_reports_unknown_share(market_db):
    r = validate.check_limit_coverage(market_db)
    assert r.blocking is False and r.passed is True
    assert 0.0 <= r.detail["unknown_share_non_suspended"] <= 1.0
    assert r.detail["by_source"]["rule"] > 0


# ══════════════ 双源交叉 ══════════════
class FakeBao:
    def __init__(self, scale=1.0, fail=False):
        self.scale, self.fail = scale, fail
    def hfq_close(self, ts_code, start, end):
        if self.fail:
            raise ConnectionError("baostock login failed")
        # 用 market_db 自己的后复权价 × scale 造对照
        c = duckdb.connect(self._path, read_only=True)
        rows = c.execute("SELECT trade_date, close * adj_factor FROM daily_bar WHERE ts_code=? "
                         "AND trade_date BETWEEN ? AND ? AND NOT is_suspended ORDER BY trade_date",
                         [ts_code, start, end]).fetchall()
        c.close()
        return pd.DataFrame(rows, columns=["trade_date", "close_hfq"]).assign(close_hfq=lambda d: d.close_hfq * self.scale)


def test_cross_source_pass_within_tolerance(market_db):
    bao = FakeBao(scale=1.001); bao._path = market_db
    r = validate.check_cross_source(market_db, bao, n_stocks=2, n_days=5)
    assert r.passed and not r.skipped and r.detail["max_abs_pct_diff"] < 0.005


def test_cross_source_fail_beyond_tolerance(market_db):
    bao = FakeBao(scale=1.02); bao._path = market_db
    r = validate.check_cross_source(market_db, bao, n_stocks=2, n_days=5)
    assert r.passed is False and r.blocking is False   # 告警级：双源差异需人工看，不阻断


def test_cross_source_skipped_when_unavailable(market_db):
    r = validate.check_cross_source(market_db, FakeBao(fail=True), n_stocks=2, n_days=5)
    assert r.skipped is True and r.passed is None


def test_validate_only_uses_read_only_connection(market_db):
    """validate.py 不得持写连接：源码级断言。"""
    import inspect
    src = inspect.getsource(validate)
    assert "connect_write" not in src


def test_row_completeness_bounds_expected_by_data_start(market_db):
    """全量回补从 2010 起：2010 前上市的股票之前没有行是正常的。期望下界 = max(list_date, 数据起点)。
    fixture 里 A 上市 2010、数据从 2023-12-25 起 → 不得误报。"""
    r = validate.check_row_completeness(market_db)
    assert r.passed and r.detail["data_start"] == D(2023, 12, 25) and r.detail["outside_window"] == 0


def test_frozen_day_blocks_promotion(market_db):
    """源某天返回空 → 全市场占位行 → 其余校验全过 → 这天若 promote 就永远不会重拉。必须阻断。"""
    _mutate(market_db, "UPDATE daily_bar SET is_suspended = TRUE, vol = 0, amount = 0, open = close, high = close, low = close "
                       "WHERE trade_date = DATE '2024-01-31'")
    r = validate.check_frozen_days(market_db)
    assert r.passed is False and r.blocking is True and r.detail["days"][0]["trade_date"] == D(2024, 1, 31)
    with pytest.raises(ValidationError) as ei:
        validate.run_all(market_db)
    assert hasattr(ei.value, "results")                 # 告警项随异常一并带出


def test_cross_source_partial_errors_do_not_discard_results(market_db):
    """单只股票查询报错 → 记录到 errors，继续比较其余；只有一只都没比上才 SKIPPED。"""
    class Flaky(FakeBao):
        def hfq_close(self, ts_code, start, end):
            if ts_code == "A00001.SZ":
                raise RuntimeError("transient")
            return super().hfq_close(ts_code, start, end)
    bao = Flaky(scale=1.0); bao._path = market_db
    r = validate.check_cross_source(market_db, bao, n_stocks=4, n_days=5)
    assert not r.skipped and r.passed and r.detail["n_errors"] == 1 and r.detail["compared"] > 0


def test_cross_source_samples_only_sh_sz_with_real_bars(market_db):
    """北交所（BaoStock 不覆盖）与窗口内无真实成交的股票不进样本。"""
    _mutate(market_db, "INSERT INTO stock_basic (ts_code, symbol, name, list_date) VALUES ('830001.BJ', '830001', '北', DATE '2020-01-01')")
    _mutate(market_db, "INSERT INTO daily_bar (ts_code, trade_date, close, adj_factor, is_suspended) VALUES ('830001.BJ', DATE '2024-02-02', 1.0, 1.0, FALSE)")
    seen = []
    class Spy(FakeBao):
        def hfq_close(self, ts_code, start, end):
            seen.append(ts_code); return super().hfq_close(ts_code, start, end)
    bao = Spy(); bao._path = market_db
    validate.check_cross_source(market_db, bao, n_stocks=10, n_days=5)
    assert "830001.BJ" not in seen and "C00003.SH" not in seen      # C 01-24 退市，窗口（最近 5 日）内无成交


def test_industry_source_downgrade_is_blocking(market_db):
    """降级来源 = 今天的行业回填到上市日 → 行业中性化前视。必须阻断，逼操作员显式承认。"""
    _mutate(market_db, "UPDATE _meta SET value='tushare_static' WHERE key='industry_source'")
    r = validate.check_industry_source(market_db)
    assert r.passed is False and r.blocking is True and r.detail["industry_source"] == "tushare_static"
    _mutate(market_db, "INSERT INTO _meta VALUES ('industry_source_ack', '1')")
    assert validate.check_industry_source(market_db).passed is True     # 显式承认后放行，键留在库里
    _mutate(market_db, "UPDATE _meta SET value='sw' WHERE key='industry_source'")
    assert validate.check_industry_source(market_db).passed is True


def test_zero_volume_non_suspended_row_is_blocking(market_db):
    _mutate(market_db, "UPDATE daily_bar SET vol = 0 WHERE ts_code='B00002.SZ' AND trade_date=DATE '2024-01-10'")
    r = validate.check_zero_volume_rows(market_db)
    assert r.passed is False and r.blocking is True and r.detail["n"] == 1
    assert r.detail["sample"][0] == {"ts_code": "B00002.SZ", "trade_date": D(2024, 1, 10)}
