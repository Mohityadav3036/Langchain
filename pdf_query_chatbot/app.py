# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# # from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_objectbox.vectorstores import ObjectBox

# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from dotenv import load_dotenv
# load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")


# st.title("ObjectBox VectorstoreDB with Llama3 Demo")

# llm = ChatGroq(groq_api_key=groq_api_key,
#                model_name="Llama3-8b-8192")

# prompts = ChatPromptTemplate.from_template(
#     """
#     Answer The questions based on th provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     </context>
#     Question: {input}

#     """
# )

# #  Vectore Embedding And Object Vectorestore DB

# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         st.session_state.loader=PyPDFDirectoryLoader("./us_census") 
#         st.session_state.docs = st.session_state.loader.load()
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
#         st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,embedding_dimensions=768)



# input_prompt = st.text_input("Enter Your Question From Documents")

# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Object Box Data bases ready")


# import time

# if input_prompt:
#     # Ensure vectors are initialized
#     if "vectors" not in st.session_state:
#         st.error("Please click 'Documents Embedding' to initialize the vector database.")
#     else:
#         try:
#             document_chain = create_stuff_documents_chain(llm, prompts)
#             retriever = st.session_state.vectors.as_retriever()
#             retriever_chain = create_retrieval_chain(retriever, document_chain)
#             start = time.process_time()

#             response = retriever_chain.invoke({'input': input_prompt})

#             st.write(f"Response time: {time.process_time() - start} seconds")
#             st.write(response['answer'])

#             # Display document similarity search results
#             with st.expander("Document Similarity Search"):
#                 for i, doc in enumerate(response["context"]):
#                     st.write(doc.page_content)
#                     st.write("--------------------")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")




import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma  # Replace ObjectBox with Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set API keys
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit app title
st.title("Chroma VectorstoreDB with Llama3 Demo")

# Initialize ChatGroq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompts = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Function to handle vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        # Initialize the Chroma vector store
        st.session_state.vectors = Chroma.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            persist_directory="./chroma_db"  # Optional: Specify a directory for persistent storage
        )

# Input prompt from user
input_prompt = st.text_input("Enter Your Question From Documents")

# Button to trigger document embedding
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Chroma Database is ready")

# Process the input prompt
if input_prompt:
    # Ensure vectors are initialized
    if "vectors" not in st.session_state:
        st.error("Please click 'Documents Embedding' to initialize the vector database.")
    else:
        try:
            document_chain = create_stuff_documents_chain(llm, prompts)
            retriever = st.session_state.vectors.as_retriever()
            retriever_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()

            response = retriever_chain.invoke({'input': input_prompt})

            st.write(f"Response time: {time.process_time() - start} seconds")
            st.write(response['answer'])

            # Display document similarity search results
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------")
        except Exception as e:
            st.error(f"An error occurred: {e}")