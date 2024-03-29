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
# hello world
result = llm.invoke("langsmith如何帮助测试?")
pprint(result)
