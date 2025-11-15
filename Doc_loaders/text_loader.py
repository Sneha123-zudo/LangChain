from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader=TextLoader(r'C:\Users\user\Desktop\LangChain\Doc_loaders\cricket.txt',encoding='utf-8')
docs = loader.load()
for doc in docs:
    print(doc.page_content)

print(type(docs))
print(len(docs))
print(docs[0])

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template="Write a summary on the cricket poem \n{poem}",
    input_variables = ['poem']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'poem':docs[0].page_content})

print("Generated Summary: " , result)