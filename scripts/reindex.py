#知识库索引重建脚本（第1步·检索纵深）
#【新增】把"手工跑 __main__ 建索引"升级为正式的运维入口——知识库更新后执行一次即可。
#用法（cd 项目根）：
#  .venv/Scripts/python.exe -m scripts.reindex            # 增量：只处理新文件/变更文件（日常用）
#  .venv/Scripts/python.exe -m scripts.reindex --rebuild  # 全量重建（换 embedding 模型/切分策略后必须）
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


# ============================================================================================
# 【第 1 步 · 1.1 说明】索引重建入口（本文件的作用与用法）
# --------------------------------------------------------------------------------------------
# 为什么需要它：
#   改造前 load_document() 全项目没有调用点，索引靠开发者手跑模块 __main__ 构建；
#   知识库文件更新后没有任何标准操作，"增量更新"这个能力等于不存在。
#   本脚本把建索引变成一条可重复执行的命令，并输出可核对的统计。
# 两种模式：
#   增量（默认）：按文件 md5 只处理新文件/内容变化的文件；内容级去重保证重复内容不入库；
#                 日常知识库更新用这个，秒级完成。
#   全量（--rebuild）：先清空 index.pkl / index_meta.json / chunks.db / md5.text 再导入；
#                 换 embedding 模型、改切分策略、怀疑索引损坏时用它。
# 输出怎么读：
#   "新增 chunk / 内容去重丢弃"——去重丢弃数明显大于 0 时，多为 PDF 与 TXT 的重叠内容
#   （属预期：同一知识只在账本里保留 TXT 版本一份）；
#   "账本总量 vs 向量索引"——正常应相等（或索引略大，见 vector_store.py 底部"已知取舍"）。
# 与版本检测的配合：
#   重建会写入 index_meta.json（embedding 模型 + 切分配置 + 结构版本）；
#   之后服务启动时若发现模型/版本不符，会直接报错并提示再来一次 --rebuild——
#   这样"换模型忘了重建"的静默故障从流程上被消除。
# ============================================================================================
