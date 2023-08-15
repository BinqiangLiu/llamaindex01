import os
import streamlit as st
import openai
openai.api_key = ""

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents from a directory
documents = SimpleDirectoryReader('data').load_data()

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

# Load in a specific embedding model
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

# Create a service context with the custom embedding model
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Create an index using the service context
new_index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
)
new_query_engine = new_index.as_query_engine()

while True:
    question = input("Your question(Enter exit to quit):\n")
    if question=="exit":
        break
    response = new_query_engine.query(question)
    print(response)


