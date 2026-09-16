import os
from langchain_core.documents import Document
# 【修改】原使用 langchain_chroma 的 Chroma，第二次运行时 hnsw 索引加载失败；
# 更换为 langchain_community 的 FAISS 向量库：本地保存索引文件，且支持对已加载索引增量追加
from langchain_community.vectorstores import FAISS
# 【修改】配置对象由 chroma_config 改为 faiss_config（见 utils/config_handler.py）
from utils.config_handler import faiss_config
from model.fectory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import txt_loader,pdf_loader,listdir_with_allowed_type
from utils.file_handler import get_file_md5_H
from utils.logger_handler import logger

class VectorStoreService:
    def __init__(self):
        # 【修改】FAISS 的持久化目录和索引文件（本工程所有向量序列化后存进这一个文件）
        self.index_dir = get_abs_path(faiss_config["persist_directory"])
        self.index_file = os.path.join(self.index_dir, "index.pkl")

        # 【修改】本地已有索引则直接加载（对应"第二次运行"场景，FAISS 加载不会报错）；
        # 首次运行时索引不存在，先置 None，等 load_document 第一次入库时用 from_documents 创建
        if os.path.exists(self.index_file):
            self._load_index()
        else:
            self.vector_store = None

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=faiss_config["chunk_size"],
            chunk_overlap=faiss_config["chunk_overlap"],
            separators=faiss_config["separators"],
            length_function = len,
        )

    # 【修改】保存索引：FAISS 自带的 save_local 走 C++ 文件读写，不支持中文路径
    # （本工程路径含中文，会报 "could not open ... for writing"），
    # 故改用 Python 原生 open 写入序列化字节，中文路径下可用
    def _save_index(self):
        os.makedirs(self.index_dir, exist_ok=True)
        with open(self.index_file, "wb") as f:
            f.write(self.vector_store.serialize_to_bytes())

    # 【修改】加载索引：同上，绕开 C++ 文件读写，用 Python 原生 open 读取后反序列化
    def _load_index(self):
        with open(self.index_file, "rb") as f:
            data = f.read()
        # allow_dangerous_deserialization=True：langchain 要求显式允许
        # 反序列化本机自己保存的 pickle 数据，加载本地索引必需
        self.vector_store = FAISS.deserialize_from_bytes(
            data,
            embed_model,
            allow_dangerous_deserialization=True,
        )

    def get_retriver(self):    #获取检索器
        # 【修改】懒加载：向量库未初始化但本地已有索引时先加载，保证单独调用检索器也能用
        if self.vector_store is None:
            if os.path.exists(self.index_file):
                self._load_index()
            else:
                logger.warning("[获取检索器]向量库尚未初始化，请先执行 load_document 加载知识库")
                return None

        # 【修改】原代码 search_kwarge 拼写错误导致 k 参数不生效，改为 search_kwargs
        return self.vector_store.as_retriever(search_kwargs={"k":faiss_config["k"]})

    def load_document(self): #从数据文件夹内读取数据文件，转为向量存入向量库，要计算文件的MD5去重，只处理没处理过的文件
        def check_md5_hex(md5_for_check:str):
            if not os.path.exists(get_abs_path(faiss_config["md5_hex_store"])):
                open(get_abs_path(faiss_config["md5_hex_store"]), "w",encoding="utf_8").close()
                return False    #md5没处理过

            with open(get_abs_path(faiss_config["md5_hex_store"]),"r",encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True    #md5处理过

                return False

        def save_md5_hex(md5_for_check:str):
            # 【修改】原代码用 "w" 覆盖写，导致只有最后一个文件的 md5 被记录、其余文件每次运行重复入库；
            # 改为 "a" 追加写，保证 md5 去重对所有文件生效
            with open(get_abs_path(faiss_config["md5_hex_store"]),"a",encoding="utf-8") as f:
                f.write(md5_for_check+"\n")

        def get_file_documents(read_path:str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)

            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)

            return[]

        allowed_files_path:list[str] =listdir_with_allowed_type(
            get_abs_path(faiss_config["data_path"]),
            tuple(faiss_config["allowed_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            #获取md5
            md5_H = get_file_md5_H(path)

            if check_md5_hex(md5_H):
                logger.info(f"[加载知识库]{path}已经存在知识库，已经跳过")
                continue

            try:
                documents:list[Document]=get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效内容，已经跳过")
                    continue

                split_document:list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效内容，已经跳过")
                    continue

                #将内容存入向量库
                # 【修改】首次入库用 from_documents 创建索引；已有索引时用 add_documents
                # 增量追加（FAISS 支持对加载后的索引追加写入，Chroma 正是在此处报 hnsw 错误）
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(split_document, embed_model)
                else:
                    self.vector_store.add_documents(split_document)

                # 【修改】每次入库后立即持久化到磁盘，避免程序退出丢数据
                self._save_index()

                #记录当已经处理好的文件的md5，避免重复
                save_md5_hex(md5_H)

                logger.info(f"[加载知识库]{path}内容加载成功")
            except Exception as e:
                #exc_info为True会记录详细的报错堆栈，如果为False仅记录报错和报错信息
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}",exc_info=True)

if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriver()
    res=retriever.invoke("键盘")
    for doc in res:
        print(doc.page_content)
        print("_"*20)
