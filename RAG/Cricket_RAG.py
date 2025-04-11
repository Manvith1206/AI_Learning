import pandas as pd
import streamlit as st
import numpy as np
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'documents' not in st.session_state:
    st.session_state.documents = None

# Initialize Google Gemini client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# File upload in sidebar
with st.sidebar:
    st.subheader("Upload Data")
    csv_file = st.file_uploader("Upload Cricket CSV", type="csv")

def convert_csv_to_nl_data(uploaded_file):
    """Convert cricket CSV data to natural language format with optimized performance"""
    if uploaded_file is None:
        st.error("Please upload a CSV file first")
        return None
        
    try:
        # Show progress
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("Loading CSV file...")
        
        # Define columns we need
        needed_columns = ['ID', 'innings', 'overs', 'ballnumber', 'batter', 'bowler', 'non-striker',
                         'extra_type', 'batsman_run', 'extras_run', 'total_run', 'isWicketDelivery',
                         'player_out', 'kind', 'fielders_involved', 'BattingTeam']
        
        # Get total rows for progress tracking (read once to count)
        df_sample = pd.read_csv(uploaded_file)
        total_rows = len(df_sample)
        uploaded_file.seek(0)  # Reset file pointer
        
        # Process in chunks for better memory usage
        chunk_size = 10000  # Adjust based on your file size
        descriptions = []
        
        # Define safe function outside the loop
        safe = lambda val: val if pd.notna(val) else "NA"
        
        # Process chunks
        processed_rows = 0
        for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
            # Pre-compute some values to avoid repetitive calculations
            chunk['over_str'] = chunk['overs'].astype(str) + '.' + chunk['ballnumber'].astype(str)
            chunk['wicket_text'] = np.where(chunk['isWicketDelivery'], 
                                          chunk.apply(lambda r: f"{safe(r['player_out'])} was dismissed ({safe(r['kind'])}). Fielders involved: {safe(r['fielders_involved'])}. ", axis=1),
                                          "")
            
            # Vectorized string building for better performance
            base_texts = "In match " + chunk['ID'].astype(str) + ", innings " + chunk['innings'].astype(str) + ", over " + chunk['over_str'] + ", " + chunk['batter'] + " faced " + chunk['bowler'] + ". "
            
            # Build run descriptions
            run_texts = []
            for _, row in chunk.iterrows():
                if pd.notna(row['extra_type']) and row['extra_type'] != "NA":
                    run_text = f"The delivery resulted in an extra ({row['extra_type']}) for {row['extras_run']} run(s). "
                elif row['batsman_run'] > 0:
                    run_text = f"The batter scored {row['batsman_run']} run(s). "
                else:
                    run_text = "No run was scored. "
                run_texts.append(run_text)
            
            # Combine all parts
            for i, row in enumerate(chunk.itertuples()):
                line = (base_texts.iloc[i] + run_texts[i] + row.wicket_text + 
                       f"Total runs from this ball: {row.total_run}. Batting team: {row.BattingTeam}.")
                descriptions.append(line)
            
            # Update progress
            processed_rows += len(chunk)
            progress_bar.progress(min(processed_rows / total_rows, 1.0))
            progress_text.text(f"Processing data... {processed_rows}/{total_rows} rows")
        
        # Write to file efficiently (single write operation)
        progress_text.text("Saving to file...")
        with open("descriptive_match_data.txt", "w") as f:
            f.write("\n".join(descriptions))
        
        progress_text.empty()
        progress_bar.empty()
        return descriptions
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None

def create_vector_store(documents):
    """Create TF-IDF vector store from documents"""
    try:
        # Create and fit TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(documents)
        
        # Store in session state
        st.session_state.vectorizer = vectorizer
        st.session_state.vector_store = vectors
        st.session_state.documents = documents
        
        # Save to disk for future use
        with open("cricket_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open("cricket_vectors.pkl", "wb") as f:
            pickle.dump(vectors, f)
        with open("cricket_documents.pkl", "wb") as f:
            pickle.dump(documents, f)
            
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def search_similar_documents(query, k=3):
    """Find similar documents using cosine similarity"""
    try:
        # Transform query using the same vectorizer
        query_vector = st.session_state.vectorizer.transform([query])
        
        # Calculate similarity with all documents
        similarity_scores = cosine_similarity(query_vector, st.session_state.vector_store).flatten()
        
        # Get top k similar documents
        top_indices = similarity_scores.argsort()[-k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "id": idx,
                "score": similarity_scores[idx],
                "content": st.session_state.documents[idx]
            })
            
        return results
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def main():
    st.title("Cricket Data RAG Analyzer")
    
    # Add sidebar for data processing
    with st.sidebar:
        st.subheader("Process Cricket Data")
        if st.button("Process Uploaded Data", key="process_data") and csv_file is not None:
            with st.spinner("Processing data..."):
                documents = convert_csv_to_nl_data(csv_file)
                if documents:
                    success = create_vector_store(documents)
                    if success:
                        st.success(f"Processed {len(documents)} cricket events")
                    else:
                        st.error("Failed to create vector store")
                        
    
    # Check if vectors are already saved
    # if (st.session_state.vector_store is None and 
    #     os.path.exists("cricket_vectorizer.pkl") and 
    #     os.path.exists("cricket_vectors.pkl") and
    #     os.path.exists("cricket_documents.pkl")):
    #     try:
    #         with open("cricket_vectorizer.pkl", "rb") as f:
    #             st.session_state.vectorizer = pickle.load(f)
    #         with open("cricket_vectors.pkl", "rb") as f:
    #             st.session_state.vector_store = pickle.load(f)
    #         with open("cricket_documents.pkl", "rb") as f:
    #             st.session_state.documents = pickle.load(f)
    #         st.info("Loaded pre-processed cricket data")
    #     except Exception as e:
    #         st.error(f"Error loading saved data: {str(e)}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat input
    if prompt := st.chat_input("Ask a question about cricket matches"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.vector_store is not None:
                    # Search for similar documents
                    similar_docs = search_similar_documents(prompt, k=3)
                    
                    # Format context for the LLM
                    context_parts = []
                    for doc in st.session_state.documents:
                        context_parts.append(f"{doc}")
                    
                    context = "\n\n".join(context_parts)
                    
                    # Generate response with Gemini
                    query = f"""
                        Behave like Cricket Analyst and be helpful cricket analyst
                        I want to analyze this ball-by-ball cricket data to understand:
                        [player performance, team strategy, match patterns, player stats in these matches]
                        Analyze all matches provided in context
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
                    st.error("Please load and process cricket data first")

if __name__ == "__main__":
    main()