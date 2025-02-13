from  langchain_huggingface import HuggingFaceEmbeddings

embedded = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text= "delhi is the capital of india"

result = embedded.embed_query(text)

print(str(result))