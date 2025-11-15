from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\user\Desktop\LangChain\Doc_loaders\dl-curriculum.pdf")
docs = loader.load()
print(docs[0].page_content)