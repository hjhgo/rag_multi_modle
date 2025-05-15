from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import torch

EMBEDDING_MODEL_NAME = "/home/hj/PycharmProjects/transfomers_d/models/jinaai/jina-embeddings-v3"
FAISS_INDEX = "faiss_index"

EMBEDDINGS = None

def get_embeddings():
    global  EMBEDDINGS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if EMBEDDINGS is None:
        try:
            EMBEDDINGS = HuggingFaceEmbeddings(
                                                   model_name=EMBEDDING_MODEL_NAME,
                                                   model_kwargs={
                                                       "device": device,
                                                       "trust_remote_code": True},
                                                   encode_kwargs={"normalize_embeddings": True}
                                               )

        except Exception as e:
            print(f"错误类型: {type(e).__name__}")
            print(f"详细错误: {str(e)}")
            # 检查 Hugging Face 缓存日志
            import traceback
            traceback.print_exc()
    return EMBEDDINGS

def build_embedding_index(texts):

    EMBEDDINGS = get_embeddings()

    db = FAISS.from_documents(texts, EMBEDDINGS)
    db.save_local(FAISS_INDEX)  # 保存索引以便复用

    return db

def build_embedding_index_form_loca():
    db = None
    try:
        EMBEDDINGS = get_embeddings()

        if os.path.isdir(FAISS_INDEX):
            db = FAISS.load_local(FAISS_INDEX,
                                  EMBEDDINGS,
                                  allow_dangerous_deserialization=True
                                  )

    except Exception as e:
        db = None

    return db