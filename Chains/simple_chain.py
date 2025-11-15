from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Generate intresting facts about \n {topic}', 
    input_variables = ['topic']
)
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'topic':'Black Hole'})
chain.get_graph().print_ascii()
print(result)