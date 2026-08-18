# scripts/check_ashare_layering.py
"""ashare/ 分层与签名静态检查（AST，不导入被检查模块）。

四类规则：
  L1 导入方向：只有 ashare/data/** 可以 import duckdb
  L2 首参名  ：ashare/data/query.py 的公开函数首参必须是 as_of_date（白名单除外）
  L3 因子签名：ashare/factors/{price,fundamental,flow,risk}.py 的公开函数前两个位置参数
               必须是 (as_of_date, universe)
  L4 写操作  ：ashare/report/**、ashare/agent_tools.py 不得出现 DML 字符串（D1）
"""
from __future__ import annotations
import ast, pathlib, sys

DUCKDB_ALLOWED_PREFIX = ("data/",)
QUERY_FIRST_PARAM_WHITELIST = {"get_tradable_mask"}     # ★ 唯一豁免，见规格 D2
READONLY_LAYERS = ("report/", "agent_tools.py")
DML = ("INSERT ", "UPDATE ", "DELETE ", "CREATE ", "DROP ", "ALTER ", "REPLACE ")
FACTOR_FILES = {"price.py", "fundamental.py", "flow.py", "risk.py"}


def _rel(path: pathlib.Path, root: pathlib.Path) -> str:
    return path.relative_to(root).as_posix()


def check(root: str = "ashare") -> list[str]:
    rootp = pathlib.Path(root)
    if not rootp.is_dir():
        # fail-closed：目录不存在也是违规，否则从别的 CWD 跑就静默"通过"
        return [f"{root}: 目录不存在（CWD={pathlib.Path.cwd()}）"]
    violations: list[str] = []

    for py in sorted(rootp.rglob("*.py")):
        rel = _rel(py, rootp)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as exc:
            violations.append(f"{rel}: 语法错误 {exc}")
            continue
        src = py.read_text(encoding="utf-8")

        # L1 导入方向
        if not rel.startswith(DUCKDB_ALLOWED_PREFIX):
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                if any(m.split(".")[0] == "duckdb" for m in mods):
                    violations.append(
                        f"{rel}:{node.lineno}: L1 直接 import duckdb —— 一切取数经 ashare/data/query.py（D2）")

        # L2 query.py 首参名
        if rel == "data/query.py":
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and not node.name.startswith("_") \
                        and node.name not in QUERY_FIRST_PARAM_WHITELIST \
                        and node.args.args:
                    first = node.args.args[0].arg
                    if first != "as_of_date":
                        violations.append(
                            f"{rel}:{node.lineno}: L2 {node.name}() 首参是 '{first}'，必须是 'as_of_date'（D2）")

        # L3 因子签名
        if rel.startswith("factors/") and py.name in FACTOR_FILES:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    names = [a.arg for a in node.args.args]
                    if names[:2] != ["as_of_date", "universe"]:
                        violations.append(
                            f"{rel}:{node.lineno}: L3 因子 {node.name}() 前两个参数必须是 "
                            f"(as_of_date, universe)，实际 {names[:2]}")

        # L4 只读层不得有 DML
        if rel.startswith(READONLY_LAYERS):
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    upper = node.value.upper().lstrip()
                    hit = next((k.strip() for k in DML if upper.startswith(k)), None)
                    if hit:
                        violations.append(
                            f"{rel}:{node.lineno}: L4 只读层出现 DML 字符串 {hit} —— 违反 D1（LLM 层无写权限）")
            if "connect_write" in src:
                violations.append(f"{rel}: L4 只读层引用了 connect_write —— 违反 D1")

    return violations


def main() -> int:
    v = check(sys.argv[1] if len(sys.argv) > 1 else "ashare")
    for line in v:
        print(f"VIOLATION  {line}")
    print(f"\n{len(v)} 处违规" if v else "\n分层检查通过")
    return 1 if v else 0


if __name__ == "__main__":
    raise SystemExit(main())
