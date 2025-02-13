from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

model = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
documents = [
    "who are you",
    "how may i help you",
    "what was the day today"
]
result = model.embed_documents(documents)

print(str(result))