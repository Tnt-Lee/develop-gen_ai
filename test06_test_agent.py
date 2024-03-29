from zhipu_tl import ChatZhipuAI
import os
from dotenv import load_dotenv
load_dotenv()
# 填写您自己的APIKey
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
llm = ChatZhipuAI(
    temperature=0.6,
    api_key=ZHIPUAI_API_KEY,
    model_name="glm-4",
)

# 创建三个自定义工具，分别是乘法工具、加法工具、指数工具。
from langchain_core.tools import tool

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)
# 引入Python代码解释器工具
from langchain_experimental.tools import PythonREPLTool
pythonREPLTool = PythonREPLTool()
# 引入维基百科查询工具
# from langchain.tools import WikipediaQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# 引入Duckduckgo搜索引擎工具
# from langchain.tools import DuckDuckGoSearchRun  -> Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead
# from langchain_community.tools import DuckDuckGoSearchRun
# search = DuckDuckGoSearchRun()  -> 相关依赖有问题
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()
search = TavilySearchResults()

# 使用structured-chat-agent提示词模板
from langchain import hub
# https://smith.langchain.com/hub
prompt = hub.pull("hwchase17/structured-chat-agent")
# 打印提示词模板
prompt.pretty_print()
# 这是一套基于ReAct的提示词模板。要想使Agent发挥作用，就必须使用到ReAct框架。

# 定义Agent使用的工具集
#定义工具
tools = [pythonREPLTool, wikipedia, search, multiply, add, exponentiate]
from langchain.agents import create_structured_chat_agent
# 创建 structured chat agent
agent = create_structured_chat_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
# 传入agent和tools来创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)
# 1）先问一个数学方面的问题

# 问一个数学问题，3的5次方乘以12和3的和，结果再平方。
# agent_executor.invoke(
#     {
#         "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
#     }
# )

agent_executor.invoke(
    {
        # "input": "美团现在的股票价格是多少？"
        # "input": "Apple stock price yesterday？"
        "input": "迈克尔乔丹生日多少?"
        # "input": "HUNTER X HUNTER是什么?"
        # "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
    }
)


#待排序的customer_list 
# customer_list = [["Harrison", "Chase"], 
#                  ["Lang", "Chain"],
#                  ["Dolly", "Too"],
#                  ["Elle", "Elem"], 
#                  ["Geoff","Fusion"], 
#                  ["Trance","Former"],
#                  ["Jen","Ayai"]
#                 ]
# agent_executor.invoke(
#     {
#         "input": f"""Sort these customers by \
#                     last name and then first name \
#                     and print the output: {customer_list}"""
#     }
# )
