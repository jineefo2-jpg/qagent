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
    assert set(m) >= {"row_completeness", "placeholder_rows", "adj_factor_jumps", "financial_ann_date",
                      "macro_publish_date", "limit_coverage", "cross_source"}
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
