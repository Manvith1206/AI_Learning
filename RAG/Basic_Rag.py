import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from google import genai

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
# Initialize session state for vector store if it doesn't exist
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    with st.spinner("Uploading Docs..."):
        uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "csv", "txt", "docx"], accept_multiple_files=True)
    with st.spinner("Processing Docs..."):
        if st.button("Process Documents"):
            documents = []
            
            for file in uploaded_files:
                file_path = f"temp_{file.name}"
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                    
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif file.name.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif file.name.endswith(".csv"):
                    loader = CSVLoader(file_path)
                else:
                    os.remove(file_path)
                    continue
                    
                documents.extend(loader.load())
                os.remove(file_path)

            if documents:
                texts = RecursiveCharacterTextSplitter.split_documents(text_splitter, documents)
                st.session_state.vector_store = FAISS.from_documents(texts, embedding=embedding)
                st.success(f"Documents processed and vector store created {st.session_state.vector_store}")
                st.write(f"Chunks Generated {len(texts)} chunks")
            
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
            else:
                st.error("Please upload and process documents first.")