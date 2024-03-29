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
# 输出解析器也是LangChain的一大提效法宝。
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
result = chain.invoke({"input": "langsmith如何帮助测试?"})
print(result)