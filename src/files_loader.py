
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter



CHUNK_SIZE = 1000  # 分块大小（字符数）
CHUNK_OVERLAP = 200  # 块间重叠字符数

# 1. 批量加载多格式文件
def load_documents(data_dir):

    # 为不同文件类型配置专用加载器
    loader_config = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader,
    }

    # 创建DirectoryLoader实例
    loader = DirectoryLoader(
        path=data_dir,
        glob="**/*.*",  # 匹配所有文件
        loader_kwargs={"mode": "elements"},  # 对复杂文档进行元素级解析
        use_multithreading=True,  # 启用多线程加速
        show_progress=True,  # 显示进度条
        silent_errors=True,  # 跳过无法解析的文件
        # loaders=loader_config,
    )

    # 执行加载
    documents = loader.load()
    print(f"成功加载 {len(documents)} 个文档")
    return documents


# 2. 智能分块处理
def split_documents(docs):

    # 创建递归字符分块器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )

    # 执行分块
    chunks = text_splitter.split_documents(docs)
    print(f"生成 {len(chunks)} 个文本块")
    return chunks


def get_files_chunks(file_path:str=""):

    documents = load_documents(data_dir=file_path)

    chunks = split_documents(documents)

    return chunks