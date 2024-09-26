import pickle
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_core.documents import Document
# from uuid import uuid4
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# Define FastAPI app
app = FastAPI()

# Global variables to store loaded data
vector_store = None
ensemble_retriever = None

# Load required files (done once on startup)
def load_files():
    with open("all_rawtexts.pkl", "rb") as filehandler:
        all_rawtexts = pickle.load(filehandler)
    with open("document_list.pkl", "rb") as filehandler:
        document_list = pickle.load(filehandler)
    with open("uidlist.pkl", "rb") as filehandler:
        uidlist = pickle.load(filehandler)
    print("loading done")
    return all_rawtexts, document_list, uidlist

# Calling OpenAI ChatGPT API for question answering
def query_openai_api(query: str, context: str):
    prompt = f"User question: {query}\n\nRelevant context: {context}\n\nAnswer based on the context:"
    client = OpenAI(
    # This is the default and can be omitted
    api_key="xx",
)
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4", depending on your access
        messages=[
            {"role": "system", "content": "You are a helpful assistant for the website PartSelect that answers questions based on provided context. Don't mention anything about the context when answering"},
            {"role": "user", "content": prompt}
        ]
    )
    return response

# Function to initialize retrievers (done once on startup)
def initialize_retrievers():
    global vector_store, ensemble_retriever

    all_rawtexts, document_list, uidlist = load_files()

    # Set up embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize FAISS index
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Add documents to the vector store
    vector_store.add_documents(documents=document_list, ids=uidlist)

    # Create BM25 and FAISS retrievers
    retriever_bm25 = BM25Retriever.from_documents(document_list)
    retriever_bm25.k = 2
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Ensemble the retrievers
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_bm25, faiss_retriever], weights=[0.5, 0.5])

    print("Retrievers initialized")

# Run this function when the app starts
@app.on_event("startup")
def on_startup():
    initialize_retrievers()

# Pydantic model for input validation
class QueryInput(BaseModel):
    query: str

# Main API endpoint for querying
@app.post("/query")
async def query_api(query_input: QueryInput):
    global ensemble_retriever
    query_text = query_input.query

    if ensemble_retriever is None:
        return {"error": "Retrievers not initialized"}

    # Retrieve documents based on the query
    docs = ensemble_retriever.invoke(query_text)
    top_docs = [i.page_content for i in docs]

    context = "\n\n".join(top_docs)

    openai_response = query_openai_api(query_text, context)
    # print(openai_response.choices[0].message.content)
    # Process and return the document content as a response
    return {"result": openai_response.choices[0].message.content}
