from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

_MODEL_NAME_ = "/home/Qwen2___5-72B-Instruct-GPTQ-Int4"
OPENAI_API_BASE = "http://192.168.59.104:31245/v1"
OPENAI_API_KEY = "EMPTY"

def build_rag_chain(retriever):

    llm = ChatOpenAI(
                        model = _MODEL_NAME_,
                        base_url = OPENAI_API_BASE,
                        api_key  = OPENAI_API_KEY,
                        temperature = 0.5
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # 返回参考文档
    )

    return qa_chain

def build_llm_chain():
    pass