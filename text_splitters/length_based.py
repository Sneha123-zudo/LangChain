from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader(r'C:\Users\user\Desktop\LangChain\Doc_loaders\cricket.txt',encoding='utf-8')
docs = loader.load()
text = docs[0]
spiltter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

result = spiltter.split_text(text.page_content)

print(result)