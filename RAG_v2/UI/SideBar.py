import streamlit as st
from concurrent.futures import ThreadPoolExecutor

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