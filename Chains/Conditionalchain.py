from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
llm1 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm1)
parser = StrOutputParser()

class Feedback(BaseModel):
    Sentiment = Literal['positive','negative'] = Field(description='Give teh sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feeback into positive or negative \n{feedback}\n{format_instruction}',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction': parser2.get_format_instruction()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'Write an appropriate feedback to this positive response \n{feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template = "Write an appropriate feedback to this negative response \n{feedback}",
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser2),
    RunnableLambda(lambda x: 'could not find sentiment')

)
chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':'This is a terrible phone'}))