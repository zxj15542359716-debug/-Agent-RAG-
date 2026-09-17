#知识库索引重建脚本
import argparse
import sys

from rag.vector_store import VectorStoreService
from utils.logger_handler import logger


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库索引重建（增量/全量）")
    parser.add_argument("--rebuild", action="store_true",
                        help="清空索引/账本/去重记录后全量重建")
    args = parser.parse_args()

    vs = VectorStoreService()
    try:
        stats = vs.rebuild() if args.rebuild else vs.load_document()
    except RuntimeError as e:
        # 版本校验失败/索引状态不一致等：错误信息本身已是可执行的修复指引
        print(f"[失败] {e}")
        return 1
    except Exception as e:
        logger.error(f"[reindex]执行失败：{e}", exc_info=True)
        print(f"[失败] {e}（详细堆栈见 logs/agent.log）")
        return 1

    print("=" * 62)
    print(("全量重建" if args.rebuild else "增量更新") + "完成")
    print(f"  处理文件 : {stats['files']} 个（跳过已处理 {stats['skipped_files']} 个）")
    print(f"  新增 chunk: {stats['chunks']} 条（内容去重丢弃 {stats['dropped_dup']} 条）")
    print(f"  账本总量 : {vs.chunk_store.count()} 条")
    print(f"  向量索引 : {vs.index_size()} 条")
    print("  各来源分布：")
    for src, n in vs.chunk_store.stats_by_source().items():
        print(f"    - {src}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
