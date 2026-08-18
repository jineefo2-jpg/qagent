from __future__ import annotations
import pathlib, sys, textwrap
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import check_ashare_layering as chk


def test_real_codebase_passes():
    """真实 ashare/ 必须零违规。这是每个任务的收尾闸门。
    用绝对路径：相对路径会让"从别的 CWD 启动 pytest"变成静默通过。"""
    assert chk.check(str(REPO / "ashare")) == []


def test_missing_root_is_a_violation(tmp_path):
    """闸门必须 fail-closed：目录不存在 ≠ 干净。"""
    v = chk.check(str(tmp_path / "nope"))
    assert v and "目录不存在" in v[0]


def _write(tmp_path, rel, src):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src))
    return p


def test_detects_direct_duckdb_import(tmp_path):
    _write(tmp_path, "factors/price.py", "import duckdb\n")
    v = chk.check(str(tmp_path))
    assert any("duckdb" in x for x in v)


def test_allows_duckdb_in_data_layer(tmp_path):
    _write(tmp_path, "data/_db.py", "import duckdb\n")
    assert chk.check(str(tmp_path)) == []


def test_detects_wrong_first_param_in_query(tmp_path):
    _write(tmp_path, "data/query.py", "def get_bars(date, codes): ...\n")
    v = chk.check(str(tmp_path))
    assert any("as_of_date" in x for x in v)


def test_allows_whitelisted_get_tradable_mask(tmp_path):
    _write(tmp_path, "data/query.py", "def get_tradable_mask(exec_date, ts_codes): ...\n")
    assert chk.check(str(tmp_path)) == []


def test_detects_write_in_report_layer(tmp_path):
    """D1：LLM 层（report/、agent_tools.py）不得出现写操作。"""
    _write(tmp_path, "report/stock_deep.py", "def f(c):\n    c.execute('INSERT INTO x VALUES (1)')\n")
    v = chk.check(str(tmp_path))
    assert any("INSERT" in x.upper() for x in v)
