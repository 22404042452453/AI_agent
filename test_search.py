#!/usr/bin/env python3
"""
Test script to verify improved GOST search functionality
"""

import os
import sys
from main import load_documents_from_directory, create_vectorstore
from search_handler import setup_search_chain

def test_search_quality():
    """Test the improved search functionality"""
    print("Testing improved GOST search functionality...")

    # Load documents
    docs_dir = "files"
    print(f"Loading documents from {docs_dir}...")
    documents = load_documents_from_directory(docs_dir)

    if not documents:
        print("No documents found!")
        return

    print(f"Loaded {len(documents)} documents")

    # Create vectorstore
    print("Creating vectorstore with improved chunking...")
    vectorstore = create_vectorstore(documents)

    # Setup search chain
    print("Setting up search chain...")
    search_chain = setup_search_chain(vectorstore)

    # Test queries
    test_queries = [
        "Какие требования к заземлению в электроустановках?",
        "Что такое коэффициент мощности?",
        "Какие нормы освещенности для производственных помещений?",
        "Требования к молниезащите зданий"
    ]

    print("\n" + "="*50)
    print("TESTING SEARCH QUALITY")
    print("="*50)

    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)

        try:
            # Get raw docs first to check metadata
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            raw_docs = retriever.get_relevant_documents(query)

            print("Retrieved chunks with metadata:")
            for j, doc in enumerate(raw_docs, 1):
                metadata = doc.metadata
                sections = metadata.get('sections', [])
                filename = metadata.get('filename', 'Unknown')
                print(f"  Chunk {j}: {filename}")
                if sections:
                    print(f"    Sections: {', '.join(sections)}")
                print(f"    Content preview: {doc.page_content[:100]}...")

            # Now get the search result
            answer = search_chain.invoke(query)
            print(f"\nSearch Answer:\n{answer}")

        except Exception as e:
            print(f"Error testing query '{query}': {e}")

        print("\n" + "-"*50)

if __name__ == "__main__":
    test_search_quality()
