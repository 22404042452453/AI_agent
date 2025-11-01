#!/usr/bin/env python3
import os
import sys
from main import load_documents_from_directory, create_vectorstore
from search_handler import setup_search_chain

# Load documents
docs_dir = "files"
print(f"Loading documents from {docs_dir}...")
documents = load_documents_from_directory(docs_dir)

if not documents:
    print("No documents found!")
    sys.exit(1)

print(f"Loaded {len(documents)} documents")

# Create vectorstore
print("Creating vectorstore...")
vectorstore = create_vectorstore(documents)

# Setup search chain
print("Setting up search chain...")
search_chain = setup_search_chain(vectorstore)

# Test query
query = " Какие защиты предусматриваются для защиты трансформатора 110 кВ"
print(f"\nTest query: {query}")

try:
    answer = search_chain.invoke(query)
    print(f"Answer:\n{answer}")
except Exception as e:
    print(f"Error: {e}")
