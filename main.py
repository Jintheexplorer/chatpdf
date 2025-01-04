import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variables (e.g., OpenAI API Key)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure OpenAI API key is available
if not openai_api_key:
    raise ValueError("Missing OpenAI API Key. Please set 'OPENAI_API_KEY' in your .env file.")

# Streamlit App UI
st.title("ChatPDF with LangChain and ChromaDB")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_question = st.text_input("Ask a question about your PDF:")
answer = None

if uploaded_file and user_question:
    try:
        # Load the PDF
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()

        # Split the document into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        # Display information about the document
        st.write(f"Loaded {len(documents)} document(s) and split into {len(docs)} chunk(s).")

        # Use OpenAI embeddings
        embed = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Use ChromaDB in-memory backend
        vector_store = Chroma(
            collection_name="in_memory_collection",  # Collection name
            embedding_function=embed  # Embedding function
        )

        # Add documents to the vector store
        vector_store.add_documents(docs)

        # Build a retrieval-based question-answering chain
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
            retriever=retriever
        )

        # Get an answer for the user's question
        answer = qa_chain.run(user_question)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display the answer
if answer:
    st.write("Answer:")
    st.write(answer)