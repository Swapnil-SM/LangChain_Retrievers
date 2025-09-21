from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

import os
from dotenv import load_dotenv
load_dotenv()  # reads .env file


# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Initialize Gemini (Google) embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Step 3: Create Chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=documents,     # List of documents to be added to the vector store
    embedding=embedding_model, # Embedding model to convert documents into vectors
    persist_directory=None,   # None means in-memory; specify a path for persistent storage
    collection_name="my_collection"
)

# Step 4: Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) #kwargs = extra keyword arguments you pass to a function.

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)