from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  
    huggingfacehub_api_token=hf_token,
    task="text-generation",
    temperature=0.7,
    max_new_tokens=200
)


model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content='You are an helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)