from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

chatModel = ChatOpenAI(model="gpt-3.5-turbo",temperature=1.5,max_completion_tokens=30)
result = chatModel.invoke("who are you")

print(result) #we have recieved many multiple arugement and parameneter in output but our main output will be on content
print(result.content)  