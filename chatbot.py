from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chat_models import AzureChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import warnings

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 用.env文件初始化环境变量
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file


import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv

os.environ["OPENAI_API_BASE"] = 'https://api.xty.app/v1'
os.environ["OPENAI_API_KEY"] = "sk-WcfAH2nafiPHdofn995509C97d7b49DcB03bCc76989b2dD3"
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ["OPENAI_API_KEY"]
# 加载文档
pdffiles = [
    "docs/cs229_lectures/东方新能源汽车主题混合型证券投资基金2023年中期报告.PDF",
    "docs/cs229_lectures/东方新能源汽车主题混合型证券投资基金2023年第2季度报告.PDF",
    "docs/cs229_lectures/东方新能源汽车主题混合型证券投资基金2023年第3季度报告.PDF",
    "docs/cs229_lectures/东方新能源汽车主题混合型证券投资基金2023年第4季度报告.PDF",
    "docs/cs229_lectures/东方新能源汽车主题混合型证券投资基金2023年第一季度报告.PDF",
    "docs/cs229_lectures/东方新能源汽车主题混合型证券投资基金2023年年度报告.PDF"
]



docs = []
for file_path in pdffiles:
    loader = PyPDFLoader(file_path)
    docs.extend(loader.load())
 
print(f"The number of docs:{len(docs)}")
# print(docs[0])

# 文档分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
splits = text_splitter.split_documents(docs)
print(f"The number of splits:{len(splits)}")
# print(splits[0])

# 向量库存储 vector embedding
embedding = OpenAIEmbeddings()

persist_directory = 'docs/chroma/'

# 由于接口限制，每次只能传16个文本块，需要循环分批传入
# for i in range(0, len(splits), 16):
#     batch = splits[i:i+16]
#     vectordb = Chroma.from_documents(
#         documents=batch,
#         embedding=embedding,
#         persist_directory=persist_directory
#     )
# vectordb.persist()

# 已经保存到向量数据库，从数据库里读取
vectordb = Chroma.from_documents(
    documents= splits,
    embedding= embedding,
    persist_directory= persist_directory
)
vectordb.persist()

print(vectordb._collection.count())

# 检索
# define retriever
# base_retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# base_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 10})

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The meta notes down the similar ",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]
document_content_description = "Lecture notes"


from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# llm = AzureChatOpenAI(deployment_name="GPT-4", temperature=0)
self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    search_type="mmr",
    search_kwargs={'k': 7, 'fetch_k': 10},
    verbose=True
)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retriever = self_query_retriever
)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=self_query_retriever,
    # retriever=compression_retriever,
    memory=memory
)

# question = "Is probability a class topic?"
# result = qa({"question": question})
# print(result['answer'])

# question = "why are those prerequesites needed?"
# result = qa({"question": question})
# print(result)

# Output 问答系统的UI实现
from flask import Flask, request, render_template

app = Flask(__name__)  # Flask APP

@app.route('/', methods=['GET', 'POST'])

def home():
    if request.method == 'POST':
        # 接收用户输入作为问题
        question = request.form.get('question')
        # RetrievalQA链 - 读入问题，生成答案
        result = qa({"question": question})
        print(result)
        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
