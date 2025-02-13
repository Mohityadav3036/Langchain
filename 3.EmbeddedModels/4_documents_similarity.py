from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()


embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents = [
    'viral is a kohli',
    'dhoni is a ms',
]

query ='tell me about viral'


embedding_documents = embedding.embed_documents(documents)
embedding_query = embedding.embed_query(query)

score = cosine_similarity([embedding_query], embedding_documents)

index, score = sorted(list(enumerate(score)),key=lambda x:x[1])[-1]

print(documents[index])
print(score)