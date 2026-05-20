"""
重建研报知识库索引 —— 用新的 Recursive Character 分块策略重新切分所有 PDF。

为什么需要：
  旧字符级切片产生的 chunks 与新 Recursive 切片产生的 ID（md5）不一致，
  upsert 不会自动覆盖，导致库里同时存在新旧两套切片 → 召回噪声。
  本脚本先 delete 旧 collection，再全量重建。

用法：
    cd demo
    python3 scripts/reindex_research_docs.py
    python3 scripts/reindex_research_docs.py --yes   # 跳过确认
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="跳过确认提示")
    args = parser.parse_args()

    from rag.config import DOCS_DIR, COLLECTION_NAME
    from rag.indexer import index_all, get_collection

    # 1. 现状统计
    print("=" * 60)
    print("重建前统计")
    print("=" * 60)
    try:
        coll = get_collection()
        before = coll.count()
        print(f"  集合 [{COLLECTION_NAME}] 当前 chunks: {before}")
    except Exception as e:
        before = 0
        print(f"  集合状态读取失败: {e}")

    pdfs = sorted(DOCS_DIR.glob("*.pdf")) if DOCS_DIR.exists() else []
    print(f"  docs/ 下的 PDF: {len(pdfs)} 份")
    if not pdfs:
        print("\n⚠️  docs/ 下没有 PDF，没什么可重建。退出。")
        return

    # 2. 确认
    print()
    if not args.yes:
        ans = input(f"将清空 [{COLLECTION_NAME}] 并重新索引 {len(pdfs)} 份 PDF。继续？[y/N] ")
        if ans.strip().lower() != "y":
            print("已取消。")
            return

    # 3. 跑重建（index_all 自带 reset=True）
    print("\n开始重建...\n")
    index_all(reset=True)

    # 4. 重建后统计
    print()
    print("=" * 60)
    print("重建后统计")
    print("=" * 60)
    coll = get_collection()
    after = coll.count()
    print(f"  集合 chunks 数: {before} → {after}")
    if after > before:
        print(f"  ✅ 新增 {after - before} chunks（新分块更精细）")
    elif after < before:
        print(f"  ✅ 减少 {before - after} chunks（新分块更紧凑）")
    else:
        print(f"  ✅ chunks 数量一致")
    print()
    print("提示：重启 server 后即可看到新切片效果")


if __name__ == "__main__":
    main()
