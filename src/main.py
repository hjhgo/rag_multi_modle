from embeding_index import build_embedding_index, build_embedding_index_form_loca
from files_loader import get_files_chunks
from model_chain import  build_rag_chain


def init_rag_env(updata_datas=True):

    # 构建向量索引, 若不用加载
    db = None
    if not updata_datas:
        db = build_embedding_index_form_loca()

    if not db:
        chunks = get_files_chunks(file_path="/home/hj/PycharmProjects/transfomers_d/datas")
        db = build_embedding_index(chunks)

    retriever = db.as_retriever(search_kwargs=dict(k=3))

    qa_chain = build_rag_chain(retriever)

    return qa_chain


def post_one_request(
                         chain:callable,
                         query:str="你好",
                         contents=None
                     ):

    result = chain.invoke({"query":query})

    return result


if __name__ == '__main__':

    chain = init_rag_env(updata_datas=False)

    while True:
        query = input("输入:\n")
        ret = post_one_request(chain, query)
        result_str = ret["result"]
        print(f"回答：{result_str}")
        for doc in ret["source_documents"]:
            print("内容:", doc.page_content)
            print("元数据:", doc.metadata)  # 假设 metadata 是字典
            print("---")




