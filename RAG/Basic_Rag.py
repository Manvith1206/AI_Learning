import streamlit as st
import os
import sys
import pandas as pd

# Add rag_modular to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_modular'))
from rag_modular.rag_pipeline import RAGPipeline
from rag_modular.config_manager import ConfigManager

TEMP_DIR = "temp_docs"

# Initialize session state for pipeline
if "pipeline" not in st.session_state:
    config_manager = ConfigManager()
    st.session_state.pipeline = RAGPipeline(config_manager)

if "documents" not in st.session_state:
    st.session_state.documents = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for configuration and document upload
with st.sidebar:
    st.subheader("Configuration")
    
    # Create tabs for different component types
    config_tabs = st.tabs(["Text Processing", "Retrieval", "Evaluation"])
    
    with config_tabs[0]:
        st.write("**Text Processing**")
        # Chunker selection
        chunker_type = st.selectbox(
            "Chunker Type",
            options=["recursive", "sentence", "semantic"],
            index=0
        )
        
        # Chunker parameters
        if chunker_type == "recursive":
            chunk_size = st.slider("Chunk Size", 100, 10000, 600)
            chunk_overlap = st.slider("Chunk Overlap", 0, 3000, 200)
            chunker_params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        elif chunker_type == "semantic":
            min_chunk_size = st.slider("Min Chunk Size", 100, 10000, 600)
            max_chunk_size = st.slider("Max Chunk Size", 100, 10000, 600)
            similarity_threshold = st.text_area("Similarity Threshold", 0.65)
            model_name = st.selectbox("Model Name", options=['all-MiniLM-L6-v2', 'paraphrase-MiniLM-L3-v2'])

            chunker_params = {"min_chunk_size": min_chunk_size, "max_chunk_size": max_chunk_size, "similarity_threshold": float(similarity_threshold), "model_name": model_name}
        elif chunker_type == "sentece":  # sentence
            max_sentences = st.slider("Max Sentences per Chunk", 1, 20, 5)
            chunker_params = {"max_sentences": max_sentences}
            
        # Embedder selection
        embedder_type = st.selectbox(
            "Embedder Type",
            options=["tfidf", "gemini"],
            index=0
        )
        # Apply text processing config
        if st.button("Apply Text Processing", key="apply_text_proc"):
            chunker_config = {"type": chunker_type, "params": chunker_params}
            st.session_state.pipeline.update_component("chunker", chunker_config)
            embedder_config = {"type": embedder_type}
            st.session_state.pipeline.update_component("embedder", embedder_config)
            st.success("Text processing configuration updated.")
    
    with config_tabs[1]:
        st.write("**Retrieval Settings**")
        # Retriever selection
        retriever_type = st.selectbox(
            "Retriever Type",
            options=["similarity", "hybrid"],
            index=0
        )
        re_ranker_type = st.selectbox(
            "Re-ranker Type",
            options=["cosine", "llm"],
            index=0
        )
        
        if re_ranker_type == "cosine":
            re_ranker_params = {"type": "cosine"}
        else:  # llm
            re_ranker_params = {"type": "llm"}
        
        if re_ranker_params["type"] == "llm":
            model = st.selectbox("LLM Model", options=["gemini-2.0-flash", "gemini-2.5-pro"], index=0)
            re_ranker_params["model"] = model
        # Retriever parameters
        if retriever_type == "similarity":
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.0, 0.01)
            retriever_params = {"similarity_threshold": similarity_threshold}
        else:  # hybrid
            keyword_weight = st.slider("Keyword Weight", 0.0, 1.0, 0.3, 0.05)
            retriever_params = {"keyword_weight": keyword_weight}
            
        # Top-k setting
        top_k = st.slider("Top-K Documents", 1, 20, 5)
        # Apply retrieval config
        if st.button("Apply Retrieval", key="apply_retrieval"):
            retriever_config = {"type": retriever_type, "params": retriever_params, "top_k": top_k}
            reranker_config = {"type": re_ranker_type, "model": "gemini-2.0-flash"}
            st.session_state.pipeline.update_component("retriever", retriever_config)
            st.session_state.pipeline.update_component("reranker", reranker_config)
            st.success("Retrieval configuration updated.")
    
    with config_tabs[2]:
        st.write("**Evaluation Settings**")
        # Evaluator selection
        evaluator_type = st.selectbox(
            "Evaluator Type",
            options=["simple", "ragas"],
            index=0
        )
        # Apply evaluation config
        if st.button("Apply Evaluation", key="apply_evaluation"):
            evaluator_config = {"type": evaluator_type}
            st.session_state.pipeline.update_component("evaluator", evaluator_config)
            st.success("Evaluation configuration updated.")

    # Document upload
    st.subheader("Upload and Process Documents")
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "csv", "txt", "docx"],
        accept_multiple_files=False
    )
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    documents, chunks = st.session_state.pipeline.process_document(uploaded_file)
                    if documents and chunks:
                        st.session_state.documents = documents
                        st.session_state.chunks = chunks
                        st.success(f"Processed {len(documents)} chunks from document")
                    else:
                        st.warning("No valid content was extracted from the document")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    # Evaluation section
    st.subheader("Evaluation")
    ground_truth = st.text_area("Ground Truth", value="")
    if st.button("Evaluate Last Query"):
        if hasattr(st.session_state.pipeline, 'last_query'):
            with st.spinner("Evaluating..."):
                try:
                    metrics = st.session_state.pipeline.evaluate(ground_truths=ground_truth)
                    
                    # Display metrics in a nice format
                    st.write("**Evaluation Metrics:**")
                    
                    metrics_df = pd.DataFrame({
                        "Metric": list(metrics.keys()),
                        "Score": list(metrics.values())
                    })
                    st.dataframe(metrics_df)
                    
                    # Show a bar chart of metrics
                    st.bar_chart(metrics_df.set_index("Metric"))
                    
                    # Store metrics in session state
                    st.session_state.last_evaluation = metrics
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
        else:
            st.warning("No query to evaluate. Ask a question first.")
            
    # Show previous evaluation if available
    if "last_evaluation" in st.session_state:
        with st.expander("Previous Evaluation Results"):
            metrics_df = pd.DataFrame({
                "Metric": list(st.session_state.last_evaluation.keys()),
                "Score": list(st.session_state.last_evaluation.values())
            })
            st.dataframe(metrics_df)

# Main chat interface
st.subheader("Chat with your Documents")
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if st.session_state.documents:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.pipeline.query(prompt)
                    st.markdown(f"**Re-ranking Explanation:**\n{response['rerank_explanation']}")
                    st.markdown(response["answer"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"]
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.error("Please upload and process documents first.")