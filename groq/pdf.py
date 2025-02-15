import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.title("Chatgroq With Llama3 Demo")

# Initialize components
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name="Llama3-8b-8192"
    )

def initialize_vector_store():
    if st.session_state.vector_store is None:
        embeddings = load_embeddings()
        
        # Load and process documents
        loader = PyPDFDirectoryLoader("./us_census")
        docs = loader.load()[:20]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Create vector store with proper embedding handling
        st.session_state.vector_store = FAISS.from_documents(
            splits, embeddings
        )

# Initialize vector store
initialize_vector_store()

llm = load_llm()

prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    <context>
    {context}
    </context>
    Question: {input}"""
)

# User interface
question = st.text_input("Enter your question about the documents:")

if question:
    if not st.session_state.vector_store:
        st.error("Vector store not initialized. Please check documents.")
        st.stop()

    try:
        # Create retrieval chain
        retriever = st.session_state.vector_store.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Process query
        start_time = time.time()
        response = retrieval_chain.invoke({"input": question})
        
        st.write(f"**Answer**: {response['answer']}")
        st.write(f"Response time: {time.time() - start_time:.2f}s")

        with st.expander("See relevant passages"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("---")

    except Exception as e:
        st.error(f"Error processing request: {str(e)}")