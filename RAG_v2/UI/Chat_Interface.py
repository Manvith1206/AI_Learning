import streamlit as st

class Chat_Interface:
    def __init__(self, model_name, session_manager, gemini_client):
        self.model_name = model_name
        self.session_manager = session_manager
        self.client = gemini_client


    def render_chat(self):
        if not self.session_manager.has_messages():
            self.session_manager.initialize_messages()

        displayChatHistory()
        self.session_manager.initializeVectorizer()
        self.session_manager.initializeVectors()

        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            self.session_manager.add_user_message(prompt)

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Check if vectors are available
                    if hasattr(st.session_state, 'vectors') and st.session_state.vectors is not None:
                        try:
                            context = similaritySearchAndAddtoContext()
                            # Create the query for Gemini
                            query = f"""
                                You are an assistant that answers questions based on the following context. Do not make up answers.
                                Answers should be in detailed
                                
                                Context:
                                {context}
                                
                                Question: {prompt}
                                
                                Answer:
                                """
                            response = self.client.models.generate_content(model="gemini-2.0-flash", contents=query)
                            
                            # Display the response
                            st.markdown(response.text)
                            
                            # Add assistant response to chat history
                            st.session_state.add_user_message(response.text)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                    else:
                        st.error("Please upload and process documents first.")


        def displayChatHistory():
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        def similaritySearchAndAddtoContext():
            # Transform query using the same vectorizer
            query_vector = self.session_manager.getVectorizer().transform([prompt])
            
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

            return context