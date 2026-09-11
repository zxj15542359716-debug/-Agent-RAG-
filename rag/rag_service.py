#总结服务，用户提问->搜索资料->提交模型->总结回复
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import  VectorStoreService
from utils.Prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.fectory import chat_model

class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriver()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model |StrOutputParser()
        return chain

    def retriever_docs(self,query:str)->list[Document]:  #检索文档
        return self.retriever.invoke(query)

    def rag_summarize(self,query:str)->str:

        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter +=1
            context +=f"[参考资料{counter}]:参考资料:{doc.page_content} | 参考元数据:{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,    #用户查询
                "context": context,    #模型回复
            }
        )

if __name__ == "__main__":
    rag = RagSummarizeService()

    print(rag.rag_summarize("哪种键盘适合打射击游戏"))