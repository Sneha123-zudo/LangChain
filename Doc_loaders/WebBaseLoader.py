from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
url = "https://www.flipkart.com/lg-2025-model-ai-convertible-6-in-1-1-5-ton-3-star-split-dual-inverter-faster-cooling-energy-saving-viraat-mode-diet-plus-ac-white/p/itm5edb39de38dc6?pid=ACNH7SB8HTAJF85N&lid=LSTACNH7SB8HTAJF85NQQCINW&marketplace=FLIPKART&fm=neo%2Fmerchandising&iid=M_e08c4c8b-e56a-44be-ae31-4ccc64df249d_11_MRUX4Z66Q28O_MC.ACNH7SB8HTAJF85N&ppt=hp&ppn=hp&ssid=ntbtth1lm80000001763114113766&otracker=clp_pmu_v2_Air%2BConditioners_1_11.productCard.PMU_V2_LG%2B2025%2BModel%2BAI%2BConvertible%2B6-in-1%2B1.5%2BTon%2B3%2BStar%2BSplit%2BAI%2BDual%2BInverter%2Bwith%2BFaster%2BCooling%2Band%2BEnergy%2BSaving%252C%2BVIRAAT%2BMode%2Band%2BDiet%2BMode%2BPlus%2BAC%2B%2B-%2BWhite_fk-sasalele-sale-tv-and-appliances-may25-at-store_ACNH7SB8HTAJF85N_neo%2Fmerchandising_0&otracker1=clp_pmu_v2_PINNED_neo%2Fmerchandising_Air%2BConditioners_LIST_productCard_cc_1_NA_view-all&cid=ACNH7SB8HTAJF85N"
loader=WebBaseLoader(url)
docs = loader.load()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template="Answer the following questions {question} from the text \n{text}",
    input_variables = ['question','text']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'question':'What is the price of the product','text':docs[0].page_content})

print("Answer: " , result)