import streamlit as st
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from google import genai

# Constants
TEMP_DIR = "temp_docs"
MAX_WORKERS = 4  # Optimal for most systems

# Initialize components
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, 
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize session states
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Ensure clean temp directory on startup
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

def process_single_file(file):
    """Process a single file with proper error handling"""
    try:
        file_ext = os.path.splitext(file.name)[1].lower()
        file_path = os.path.join(TEMP_DIR, file.name)
        
        # Write to temp file
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Load based on file type
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_ext == '.txt':
            loader = TextLoader(file_path)
        elif file_ext == '.csv':
            loader = CSVLoader(file_path)
        else:
            return None
            
        return loader.load()
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

with st.sidebar:
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
                    # Flatten and split documents
                    all_docs = [item for sublist in documents for item in sublist]
                    texts = text_splitter.split_documents(all_docs)
                    
                    # Create vector store
                    try:
                        st.session_state.vector_store = FAISS.from_documents(
                            texts, 
                            embedding=embedding
                        )
                        st.success(f"Processed {len(texts)} chunks from {len(uploaded_files)} files")
                        st.write(f"Vector store size: {st.session_state.vector_store.index.ntotal} vectors")
                    except Exception as e:
                        st.error(f"Error creating vector store: {str(e)}")
                else:
                    st.warning("No valid documents were processed")

llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",  # Using a larger model that better supports text generation
    huggingfacehub_api_token=st.secrets["HUGGINGFACE_API_KEY"],
    model_kwargs={"temperature": 0.5},
    task="text2text-generation"  # Explicitly specifying the task
)

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
            # Debug: Display vector store status
            if st.session_state.vector_store is not None:
                # Create RAG pipeline with proper configuration
                query_embed = embedding.embed_query(prompt)
                similar_docs_with_scores = st.session_state.vector_store.similarity_search_with_relevance_scores(prompt, k=2)

                context_parts = []
                for doc, score in similar_docs_with_scores:
                    context_parts.append(f"Document ID: {doc.id} (Similarity: {score:.4f})\n{doc.page_content}")
                
                context = "\n\n".join(context_parts)

                query = f"""
                    Based on the following context, please answer the question.
                    
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
                context = context + "\n\n" + response.text
            else:
                st.error("Please upload and process documents first.")