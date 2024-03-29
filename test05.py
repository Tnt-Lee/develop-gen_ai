from zhipu_tl import ChatZhipuAI
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
# # 检索链(Retrieval Chain)
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com")

# 使用嵌入模型进行向量化，再存储到向量数据库
docs = loader.load()
# 因为OpenAIEmbeddings嵌入模型需要和OpenAI ChatGPT配套使用。
# 我们换成更通用的HuggingFaceEmbeddings --> pip install sentence-transformers
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 本地向量数据库FAISS
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# 然后在向量数据库中建立索引
vector = FAISS.from_documents(documents, embeddings)

# 创建一个检索链
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:

<context>
{context}
</context>

问题: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
# 使用检索器动态选择最相关的文档，并将其传递给检索链。
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# 3、智能体(Agent)
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "搜索有关LangSmith的信息。关于LangSmith的任何问题，您都可以使用这个工具",
)


from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()
tools = [retriever_tool, search]

from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": "langsmith如何帮助测试?"})
agent_executor.invoke({"input": "旧金山的天气怎么样?"})
