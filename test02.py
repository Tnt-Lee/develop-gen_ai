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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# 提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的技术文档作者。"),
    ("user", "{input}")
])
chain = prompt | llm
# result = chain.invoke({"input": "langsmith如何帮助测试?"})
result = chain.invoke({"input": "大语言模型如何在软件工程的垂直领域应用?"})
print(result)