from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="local-model",
    base_url="http://127.0.0.1:8000/v1", # 刚刚启动的服务
    api_key="666"                      # 本地服务随便填
)

response = llm.invoke("你会啥")
print(response.content)