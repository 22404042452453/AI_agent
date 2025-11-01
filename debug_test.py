#!/usr/bin/env python3
import os
import sys

# Try to load existing vectorstore first
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    if os.path.exists("./faiss_index"):
        print("Loading existing vectorstore...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("Vectorstore loaded successfully")
    else:
        print("No existing vectorstore found")
        sys.exit(1)
except Exception as e:
    print(f"Error loading vectorstore: {e}")
    sys.exit(1)

# Test retrieval
query = "Какие требования к заземлению в электроустановках?"
print(f"\nTest query: {query}")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
raw_docs = retriever.get_relevant_documents(query)

print("Retrieved chunks:")
for i, doc in enumerate(raw_docs, 1):
    metadata = doc.metadata
    sections = metadata.get('sections', [])
    filename = metadata.get('filename', 'Unknown')

    print(f"\nChunk {i}:")
    print(f"  Filename: {filename}")
    if sections:
        print(f"  Sections: {', '.join(sections)}")
    print(f"  Content preview: {doc.page_content[:200]}...")

# Test formatting
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata
        sections = metadata.get('sections', [])
        filename = metadata.get('filename', 'Unknown')

        # Clean up filename to get document designation
        doc_name = filename.replace('.pdf', '').replace('_', ' ').strip()

        # Add section references to the content
        if sections:
            section_refs = ", ".join(sections)
            content = f"[Разделы: {section_refs}] {content}"

        # Add document reference
        content = f"[Документ: {doc_name}]\n{content}"

        formatted_docs.append(content)

    return "\n\n".join(formatted_docs)

formatted = format_docs(raw_docs)
print(f"\nFormatted context:\n{formatted}")
