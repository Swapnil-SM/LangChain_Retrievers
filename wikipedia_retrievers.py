from langchain_community.retrievers import WikipediaRetriever

from dotenv import load_dotenv
load_dotenv()  # reads .env file



# Initialize the retriever (optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=2, lang="en") # You can change "en" to any other language code if needed
# Define your query
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

# Get relevant Wikipedia documents
docs = retriever.invoke(query)

# Print retrieved content
for i, doc in enumerate(docs): #enumerate(docs) gives both the index (i) and the document (doc) for each result returned by Wikipedia.
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  # truncate for display