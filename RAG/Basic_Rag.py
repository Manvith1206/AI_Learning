import streamlit as st
import os
import shutil
import numpy as np
import pandas as pd
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from google import genai
# For document loading
import PyPDF2
import docx2txt
import csv


# Constants
TEMP_DIR = "temp_docs"
MAX_WORKERS = 4  # Optimal for most systems
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200

# Initialize Google Generative AI client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize session states
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = TfidfVectorizer()
    
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
    
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

# Ensure clean temp directory on startup
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_text_from_file(file_path, file_ext):
    """Extract text from different file types"""
    if file_ext == '.pdf':
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    elif file_ext == '.docx':
        return docx2txt.process(file_path)
    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif file_ext == '.csv':
        text = ""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                text += " ".join(row) + "\n"
        return text
    else:
        return None

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    if not text:
        return []
        
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds chunk size, save current chunk and start new one
        print("Para: " + paragraph)
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from previous chunk
            words = current_chunk.split()
            if len(words) > chunk_overlap:
                current_chunk = " ".join(words[-chunk_overlap:]) + "\n"
            else:
                current_chunk = ""
                
        current_chunk += paragraph + "\n\n"
        
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def process_single_file(file):
    """Process a single file with proper error handling"""
    try:
        file_ext = os.path.splitext(file.name)[1].lower()
        file_path = os.path.join(TEMP_DIR, file.name)
        
        # Write to temp file
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Extract text from file
        text = extract_text_from_file(file_path, file_ext)
        if not text:
            return None
            
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        
        # Create document objects similar to LangChain format for compatibility
        documents = []
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            documents.append({
                "id": doc_id,
                "page_content": chunk,
                "metadata": {"source": file.name}
            })
            
        return documents
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

with st.sidebar:
    st.subheader("Upload and Process Documents")
    with st.spinner("Uploading Docs..."):
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=["pdf", "csv", "txt", "docx"], 
            accept_multiple_files=True
        )
    
    if uploaded_files:
        with st.spinner("Processing Docs..."):
            if st.button("Process Documents"):
                # Process files in parallel with progress
                progress_bar = st.progress(0)
                
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(process_single_file, file) for file in uploaded_files]
                    documents = []
                    
                    for i, future in enumerate(futures):
                        result = future.result()
                        if result:
                            documents.append(result)
                        progress_bar.progress((i + 1) / len(futures))
                
                if documents:
                    # Flatten documents
                    all_docs = [item for sublist in documents if sublist for item in sublist]
                    
                    if all_docs:
                        # Extract text from documents
                        texts = [doc["page_content"] for doc in all_docs]
                        st.session_state.chunks = all_docs
                        
                        # Create vector store using TF-IDF and Nearest Neighbors
                        try:
                            # Fit the vectorizer and transform documents
                            st.session_state.vectorizer = TfidfVectorizer()
                            st.session_state.vectors = st.session_state.vectorizer.fit_transform(texts)
                            
                            # Initialize nearest neighbors model
                            st.session_state.nn_model = NearestNeighbors(
                                n_neighbors=min(5, len(texts)),  # Limit to 5 or number of texts if less
                                metric='cosine'
                            )
                            st.session_state.nn_model.fit(st.session_state.vectors)
                            
                            st.success(f"Processed {len(texts)} chunks from {len(uploaded_files)} files")
                            st.write(f"Vector store size: {st.session_state.vectors.shape[0]} vectors")
                        except Exception as e:
                            st.error(f"Error creating vector store: {str(e)}")
                    else:
                        st.warning("No valid content was extracted from documents")
                else:
                    st.warning("No valid documents were processed")

# We'll use Google's Gemini model directly instead of HuggingFace

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if vectors are available
            if hasattr(st.session_state, 'vectors') and st.session_state.vectors is not None:
                try:
                    # Transform query using the same vectorizer
                    query_vector = st.session_state.vectorizer.transform([prompt])
                    
                    # Find nearest neighbors
                    distances, indices = st.session_state.nn_model.kneighbors(query_vector, n_neighbors=4)
                    
                    # Convert distances to similarity scores (1 - distance)
                    similarity_scores = 1 - distances.flatten()
                    
                    # Get the relevant documents
                    context_parts = []
                    for i, (idx, score) in enumerate(zip(indices.flatten(), similarity_scores)):
                        doc = st.session_state.chunks[idx]
                        context_parts.append(f"Document ID: {doc['id']} (Similarity: {score:.4f})\n{doc['page_content']}")
                    
                    context = "\n\n".join(context_parts)
                    
                    # Create the query for Gemini
                    query = f"""
                        You are an assistant that answers questions based on the following context. Do not make up answers.
                        Answers should be in detailed
                        
                        Context:
                        {context}
                        
                        Question: {prompt}
                        
                        Answer:
                        """
                    response = client.models.generate_content(model="gemini-2.0-flash", contents=query)
                    
                    # Display the response
                    st.markdown(response.text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            else:
                st.error("Please upload and process documents first.")