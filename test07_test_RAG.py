from zhipu_tl import ChatZhipuAI
from pprint import pprint
import os
from dotenv import load_dotenv
load_dotenv()
# 填写您自己的APIKey
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
llm = ChatZhipuAI(
    temperature=0.1,
    api_key=ZHIPUAI_API_KEY,
    model_name="glm-4",
)
# 确认当前路径
import os
# print(os.getcwd())
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path=r'C:\Users\youzi\anaconda3\envs\langchain\zhipu-TL\test07_files\ordersample.csv')
data = loader.load()
# from pprint import pprint 
# 安装不了
# print(data)
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(data)
# len(all_splits)
# 向量化存储
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bgeEmbeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
from langchain_community.vectorstores import FAISS
vector = FAISS.from_documents(all_splits, bgeEmbeddings)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
result = retriever.invoke("收货人姓名是张三丰的，有几个订单？金额分别是多少，总共是多少？")
print(result)
# 使用检索链，串联向量库检索和大模型，根据用户的提问，生成问答结果。
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:
                                            <context>
                                            {context}
                                            </context>
                                            问题: {question}""")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
retriever_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = retriever_chain.invoke("订单ID是123456的收货人是谁，电话是多少?")
print('问题1答案：')
pprint(result)
result = retriever_chain.invoke("收货人张三丰有几个订单？金额分别是多少，总共是多少？")
print('问题2答案：')
pprint(result)
print()
result = retriever_chain.invoke("收货地址是朝阳区的有哪些订单？")
print('问题3答案：')
pprint(result)
