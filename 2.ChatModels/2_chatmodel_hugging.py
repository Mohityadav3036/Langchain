from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

HUGGINGFACEHUB_API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print("Hugging Face API Key:", HUGGINGFACEHUB_API_KEY)  # Debugging step

llm = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_KEY
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of india")


print(result.content)