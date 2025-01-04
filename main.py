import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Streamlit app title and description
st.title("📚 Chat with PDF")
st.write("Upload a PDF, and ask questions about its content.")

# Step 1: File upload for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded file temporarily
    file_path = f"./{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Step 2: Load the OpenAI API key from the .env file
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("Missing OpenAI API key. Please set it in the .env file.")
            st.stop()

        # Step 3: Load the PDF file
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            st.error("No content could be loaded from the PDF file.")
            st.stop()

        st.success(f"Loaded {len(docs)} document(s) from the PDF.")

        # Step 4: Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust the chunk size as needed
            chunk_overlap=100  # Overlap for better context
        )
        split_docs = text_splitter.split_documents(docs)

        if not split_docs:
            st.error("No chunks were created. Check the input document or text splitter settings.")
            st.stop()

        st.success(f"Split into {len(split_docs)} chunk(s).")

        # Step 5: Initialize OpenAI Embeddings
        embed = OpenAIEmbeddings(
            model="text-embedding-ada-002",  # Correct OpenAI embedding model
            openai_api_key=openai_api_key   # Provide the API key
        )

        # Step 6: Store the embeddings in Chroma vector database
        vector_store = Chroma(
            collection_name="my_collection",
            embedding_function=embed,
            persist_directory="./chroma_store"  # Specify where to persist the vector store
        )

        # Add documents to the vector store
        vector_store.add_documents(split_docs)

        st.success("Documents successfully embedded and stored in Chroma vector database.")

        # Step 7: User input for question
        question = st.text_input("Ask a question about the PDF content:")

        if st.button("Get Answer"):
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 results
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)  # Use OpenAI for LLM-based answering

            # Create a RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True  # Include source documents in the response
            )

            # Query the chain
            response = qa_chain({"query": question})

            st.write("### Question:")
            st.write(question)

            st.write("### Answer:")
            st.write(response['result'])

            st.write("### Sources:")
            for doc in response['source_documents']:
                st.json(doc.metadata)

    except Exception as e:
        st.error(f"An error occurred: {e}")