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
from ragas.metrics import context_precision, context_recall, faithfulness, answer_correctness
from ragas import evaluate
from datasets import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
TEMP_DIR = "temp_docs"
MAX_WORKERS = 4  # Optimal for most systems
CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
import openai
os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_AI_API_KEY"]

# Initialize Google Generative AI client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize session states
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = TfidfVectorizer()
    
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
    
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if "query" not in st.session_state:
    st.session_state.query = None
if "query" not in st.session_state:
    st.session_state.query = None
if "context_docs" not in st.session_state:
    st.session_state.context_docs = []
if "assistant_response" not in st.session_state:
    st.session_state.assistant_response = None

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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(text)
    
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
        chunks = split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        # Create document objects similar to LangChain format for compatibility
        documents = []
        file_chunks = []
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            file_chunks.append(chunk)
            documents.append({
                "id": doc_id,
                "page_content": chunk,
                "metadata": {"source": file.name}
            })
        
        # Return both the documents and chunks for this file
        return documents, file_chunks
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None, None
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

with st.sidebar:
    result = None
    if st.button("Evaluate RAG"):
        print("Evaluate")
        def run_ragas_eval(question, answer, contexts, ground_truths=[""]):
            # Create a HuggingFace Dataset
            # print("Question: ")
            # print(type(question))
            # print("Answer: ")
            # print(type(answer))
            # print("Contexts: ")
            # print(type(contexts))

            data = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truths": [ground_truths],
            })

            result = evaluate(
                data,
                metrics=[
                    faithfulness
                ]
            )
            
            return result

        if (st.session_state.query and st.session_state.assistant_response and st.session_state.context_docs):
            result = run_ragas_eval(st.session_state.query, st.session_state.assistant_response, st.session_state.context_docs)
    if result:
        print(result["faithfulness"])
        st.text_area("Faithfulness", value=result["faithfulness"])

    st.subheader("Upload and Process Documents")
    with st.spinner("Uploading Docs..."):
        uploaded_file = st.file_uploader(
            "Upload Document", 
            type=["pdf", "csv", "txt", "docx"], 
            accept_multiple_files=False
        )
    
    if uploaded_file:
        with st.spinner("Processing Doc..."):
            if st.button("Process Document"):
                result, file_chunks = process_single_file(uploaded_file)
                if result and file_chunks:
                    # Extract text from documents
                    texts = [doc["page_content"] for doc in result]
                    st.session_state.chunks = result
                    st.session_state.context_docs = file_chunks

                    # Create vector store using TF-IDF and Nearest Neighbors
                    try:
                        st.session_state.vectorizer = TfidfVectorizer()
                        st.session_state.vectors = st.session_state.vectorizer.fit_transform(texts)

                        st.session_state.nn_model = NearestNeighbors(
                            n_neighbors=min(5, len(texts)),
                            metric='cosine'
                        )
                        st.session_state.nn_model.fit(st.session_state.vectors)

                        st.success(f"Processed {len(texts)} chunks from 1 file")
                        st.write(f"Vector store size: {st.session_state.vectors.shape[0]} vectors")
                    except Exception as e:
                        st.error(f"Error creating vector store: {str(e)}")
                else:
                    st.warning("No valid content was extracted from the document")

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
                    index = 0
                    for i in range(len(indices[0])):
                        idx = indices[0][i]
                        index+=1
                        # Get the document chunk
                        doc = st.session_state.context_docs[idx]
                        
                        # Add only the content to context parts
                        context_parts.append(doc)
                    print("Index: " + str(index))
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
                    st.session_state.query = query
                    st.session_state.assistant_response = response.text

                    print("Context: ")
                    breakpoint()
                    print(context)

                    session_state_docs = []
                    for chunk in st.session_state.context_docs:
                        session_state_docs.append(chunk)

                    # Display the response
                    st.markdown(response.text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            else:
                st.error("Please upload and process documents first.")