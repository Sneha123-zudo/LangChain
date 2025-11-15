from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
text = 'Delhi is the capital of India'
vector = embeddings.embed_query(text)
print(str(vector))