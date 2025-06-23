import infrastructure.common.rag_constants as constants
import traceback
from infrastructure.prompt_providers.llm_chat_prompt_provider import LLM_Chat_Prompt_Provider
from infrastructure.common.query_classifier_llm import QueryClassifier
from infrastructure.llm_chat_services.base_llm_service import BaseLLMService
from infrastructure.vector_stores.base_vector_store import BaseVectorStore
from infrastructure.embedders.base_embedder import BaseEmbedder
from infrastructure.retrieval_methods.base_retriever import BaseRetriever
from infrastructure.rerankers.base_reranker import BaseReranker

class QueryProcessing:
    def __init__(self, llm_service: BaseLLMService, vector_store: BaseVectorStore, embedder: BaseEmbedder,
                 retriever: BaseRetriever, reranker: BaseReranker, error_callback):
        self.query_classifier = QueryClassifier()
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.error_callback = error_callback
        
    def greetUser(self, query_text):
        if self.query_classifier.is_greeting(query_text):
            return {
                constants.ANSWER: self.query_classifier.get_greeting_response(),
                constants.CONTEXTS: "",
                constants.RERANK_EXPLANATION: ""
            }
        
    def irrelvant(self, query_text):
        context_docs = self.get_context_docs(query_text)
        if self.query_classifier.is_irrelevant(query_text, context_docs):
            return {
                constants.ANSWER: self.query_classifier.get_irrelevant_question_response(),
                constants.CONTEXTS: context_docs,
                constants.RERANK_EXPLANATION: ""
            }
        
    def get_context_docs(self, query_text):
        
        if not hasattr(self.vector_store, 'documents') or not self.vector_store.documents:
                raise ValueError("No documents processed. Please upload and process a document before querying.")
        
        # Generate query embedding
        query_embedding = self.embedder.transform([query_text])
        if isinstance(query_embedding, list) and query_embedding:
            first = query_embedding[0]
            if hasattr(first, "values"):
                query_embedding = [e.values for e in query_embedding]
            elif hasattr(first, "embedding"):
                query_embedding = [e.embedding for e in query_embedding]
                    
        results = self.retriever.retrieve(
                query_embedding, 
                self.vector_store.documents, 
                vector_store=self.vector_store,
                query_text=query_text
                )
        retrieved_docs = [result for result in results]
            
        # Use retriever to get relevant documents
        if not retrieved_docs:
            raise ValueError(constants.UNABLE_TO_RETRIEVE_MESSAGE)
        
        # Rerank documents
        reranked_docs, explanation = self.reranker.rerank(query_text, retrieved_docs)
        
        context_docs = None
        if reranked_docs:
            context_docs = "\n\n".join(reranked_docs)
            context_docs_list = reranked_docs
        else:
            context_docs = "\n\n".join(retrieved_docs)
            context_docs_list = retrieved_docs

        return context_docs, explanation, context_docs_list
    
    def query(self, query_text, history_text, top_k=None):
        try:
            print("Query")
            if self.query_classifier.is_greeting(query_text):
                # UIComponents.create_subheader_UI(self.query_classifier.get_greeting_response())
                # UIComponents.add_message_to_chat("assistant",  self.query_classifier.get_greeting_response())
                yield {
                constants.ANSWER: self.query_classifier.get_greeting_response(),
                constants.CONTEXTS: "",
                constants.RERANK_EXPLANATION: ""
            }
                return
            # query_text = self.rewrite_query(query_text)
            # Ensure documents are available
            
            context_docs, explanation, context_docs_list = self.get_context_docs(query_text)
            if self.query_classifier.is_irrelevant(query_text, context_docs):
                # UIComponents.create_subheader_UI(self.query_classifier.get_irrelevant_question_response())
                # UIComponents.add_message_to_chat("assistant",  self.query_classifier.get_irrelevant_question_response())
                yield {
                constants.ANSWER: self.query_classifier.get_irrelevant_question_response(),
                constants.CONTEXTS: context_docs,
                constants.RERANK_EXPLANATION: ""
            }
                return
            # Join contexts
            context = "\n\n".join(context_docs)
            with open("Contexts.txt", "w", encoding="utf-8") as file:
                file.write(context)

            llm_chat_prompt_provider = LLM_Chat_Prompt_Provider()
            # Generate answer
            answer_prompt = llm_chat_prompt_provider.get_final_prompt(context=context, query_text=query_text, history_text=history_text)
            print("AnswerPrompt: ", answer_prompt)
            full_response = ""
            for delta in self.llm_service.generate_response(answer_prompt):
                full_response += delta
                yield {
                constants.ANSWER: full_response,
                constants.CONTEXTS: context_docs,
                constants.RERANK_EXPLANATION: ""
            }

            # Save the query data for potential evaluation
            self.last_query = {
                constants.QUESTION: query_text,
                constants.ANSWER: full_response,
                constants.CONTEXTS: context_docs_list
            }
            
        except Exception as e:
            self.error_callback(f"Error during query: {e}, Traceback: {traceback.print_exc()}")
            return None