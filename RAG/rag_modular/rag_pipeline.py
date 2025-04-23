import os
import uuid
from .recursive_chunker import RecursiveChunker
from .sentence_chunker import SentenceChunker
from .tfidf_embedder import TFIDFEmbedder
from .gemini_embedder import GeminiEmbedder
from .sklearn_vector_store import SklearnVectorStore
from .llm_reranker import LLMReranker
from .gemini_service import GeminiService
from .similarity_retriever import SimilarityRetriever
from .hybrid_retriever import HybridRetriever
from .simple_evaluator import SimpleEvaluator
from .ragas_evaluator import RagasEvaluator
from .config_manager import ConfigManager
import streamlit as st
from .cosine_reranker import CosineReranker
from .semantic_chunker import SemanticChunker

class RAGPipeline:
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or ConfigManager()
        self.setup_components()

    def setup_components(self):
        # Chunker
        chunker_config = self.config_manager.get_config("chunker")
        if chunker_config["type"] == "recursive":
            self.chunker = RecursiveChunker(**chunker_config.get("params", {}))
        elif chunker_config["type"] == "sentence":
            self.chunker = SentenceChunker(**chunker_config.get("params", {}))
        elif chunker_config["type"] == "semantic":
            self.chunker = SemanticChunker(**chunker_config.get("params", {}))
        else:
            self.chunker = RecursiveChunker()
            
        # Embedder
        embedder_config = self.config_manager.get_config("embedder")
        if embedder_config["type"] == "tfidf":
            self.embedder = TFIDFEmbedder()
        elif embedder_config["type"] == "gemini":
            self.embedder = GeminiEmbedder(api_key=st.secrets["GEMINI_API_KEY"])
        else:
            self.embedder = TFIDFEmbedder()
            
        # Vector Store
        vector_store_config = self.config_manager.get_config("vector_store")
        self.vector_store = SklearnVectorStore(**vector_store_config.get("params", {}))
        
        # Retriever
        retriever_config = self.config_manager.get_config("retriever")
        if retriever_config["type"] == "similarity":
            self.retriever = SimilarityRetriever(**retriever_config.get("params", {}))
        elif retriever_config["type"] == "hybrid":
            self.retriever = HybridRetriever(**retriever_config.get("params", {}))
        else:
            self.retriever = SimilarityRetriever()
        self.top_k = retriever_config.get("top_k", 5)
        
        # LLM Service
        llm_config = self.config_manager.get_config("llm")
        if llm_config["type"] == "gemini":
            from google import genai
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            self.llm_service = GeminiService(client, model_name=llm_config.get("model", "gemini-2.0-flash"))
        else:
            from google import genai
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            self.llm_service = GeminiService(client)
            
        # Reranker
        reranker_config = self.config_manager.get_config("reranker")
        # Always use LLM Reranker
        if reranker_config["type"] == "llm":
            self.reranker = LLMReranker(
                self.llm_service.client,
                model_name=reranker_config.get("model", "gemini-2.0-flash")
            )
        elif reranker_config["type"] == "cosine":
            self.reranker = CosineReranker(self.embedder)
        
        # Evaluator
        evaluator_config = self.config_manager.get_config("evaluator")
        if evaluator_config["type"] == "ragas":
            self.evaluator = RagasEvaluator(**evaluator_config.get("params", {}))
        elif evaluator_config["type"] == "simple":
            self.evaluator = SimpleEvaluator()
        else:
            self.evaluator = SimpleEvaluator()

    def process_document(self, file, temp_dir="temp_docs"):
        from .document_loaders.pdf_loader import PDFLoader
        from .document_loaders.docx_loader import DOCXLoader
        from .document_loaders.txt_loader import TXTLoader
        from .document_loaders.csv_loader import CSVLoader
        loaders = {
            '.pdf': PDFLoader(),
            '.docx': DOCXLoader(),
            '.txt': TXTLoader(),
            '.csv': CSVLoader(),
        }
        try:
            os.makedirs(temp_dir, exist_ok=True)
            file_ext = os.path.splitext(file.name)[1].lower()
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            print("File saved successfully")
            if file_ext in loaders:
                text = loaders[file_ext].load_document(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            print("Documents loaded succesfully")
            if not text:
                return None, None
            print("Text loaded successfully")
            chunks =  self.chunker.split_text(text=text)


            print("Chunks generated successfully")
            breakpoint()
            documents = []
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                documents.append({
                    "id": doc_id,
                    "page_content": chunk,
                    "metadata": {"source": file.name}
                })
            print("Docs loaded from chunks data")
            texts = [doc["page_content"] for doc in documents]
            print("Texts extracted successfully")
            embeddings = self.embedder.fit(texts)
            print("Embeddings generated successfully")
            print(type(embeddings))
            print(type(documents))
            
            self.vector_store.add_embeddings(embeddings, documents)
            print("Self docs: ")
            print(self.vector_store.documents[0])
            print("Documents processed successfully")
            return documents, chunks
        except Exception as e:
            st.error(f"Error processing document: in Process Doc Function {str(e)}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def query(self, query_text, top_k=None):
        print("Query Started")
        # Ensure documents are available
        if not hasattr(self.vector_store, 'documents') or not self.vector_store.documents:
            raise ValueError("No documents processed. Please upload and process a document before querying.")
        print("Querying...")
        # Use configured top_k if not specified
        if top_k is None:
            top_k = self.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.transform([query_text])
        print("Query embedding generated successfully")
        
        # Use retriever to get relevant documents
        results = self.retriever.retrieve(
            query_embedding, 
            self.vector_store.documents, 
            top_k=top_k,
            vector_store=self.vector_store,
            query_text=query_text
        )
        print("Results retrieved successfully")
        
        # Extract document content
        retrieved_docs = [result["document"]["page_content"] for result in results]
        print("Docs retrieved successfully")
        print(retrieved_docs)
        
        # Rerank documents
        reranked_docs, explanation = self.reranker.rerank(query_text, retrieved_docs)
        print("Docs reranked successfully")
        
        context_docs = None
        if reranked_docs:
            context_docs = "\n\n".join(reranked_docs)
        else:
            context_docs = "\n\n".join(retrieved_docs)

        # Join contexts
        context = "\n\n".join(context_docs)
        print("Context generated successfully")
        
        # Generate answer
        answer_prompt = f"""
            You are an assistant that answers questions based on the following context. Do not make up answers.
            Answers should be detailed.

            Context:
            {context}

            Question: {query_text}

            Answer:
            """
        answer = self.llm_service.generate_response(answer_prompt)
        
        # Save the query data for potential evaluation
        self.last_query = {
            "question": query_text,
            "answer": answer,
            "contexts": retrieved_docs
        }
        
        
        return {
            "answer": answer,
            "context": context_docs,
            "rerank_explanation": explanation
        }
        
    def evaluate(self, question=None, answer=None, contexts=None, ground_truths=None):
        """Evaluate the RAG system using the configured evaluator
        
        Args:
            question: The question to evaluate (uses last query if None)
            answer: The answer to evaluate (uses last query if None)
            contexts: The contexts to evaluate (uses last query if None)
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        # Use last query data if not provided
        if hasattr(self, 'last_query') and (question is None or answer is None or contexts is None):
            question = question or self.last_query["question"]
            answer = answer or self.last_query["answer"]
            contexts = contexts or self.last_query["contexts"]

        eval_questions = []
        with open('RAG/generated_questions.text', 'r') as file:
            for line in file:
                # Remove newline character and convert to integer
                item = line.strip()
                eval_questions.append(item)

        if not (question and answer and contexts):
            raise ValueError("No query data available for evaluation")
        
        # Run evaluation
        metrics = self.evaluator.evaluate(question, answer, contexts, ground_truths)
        return metrics

    def update_component(self, component_name, config):
        self.config_manager.update_config(component_name, config)
        # Only rebuild components that require reinitialization
        if component_name in ["chunker", "embedder", "vector_store", "llm", "reranker"]:
            # these components affect document processing, re-setup entire pipeline
            self.setup_components()
        elif component_name == "retriever":
            # update retriever only
            retriever_config = self.config_manager.get_config("retriever")
            if retriever_config.get("type") == "similarity":
                self.retriever = SimilarityRetriever(**retriever_config.get("params", {}))
            elif retriever_config.get("type") == "hybrid":
                self.retriever = HybridRetriever(**retriever_config.get("params", {}))
            else:
                self.retriever = SimilarityRetriever()
            self.top_k = retriever_config.get("top_k", self.top_k)
        elif component_name == "evaluator":
            # update evaluator only
            evaluator_config = self.config_manager.get_config("evaluator")
            if evaluator_config.get("type") == "ragas":
                self.evaluator = RagasEvaluator(**evaluator_config.get("params", {}))
            else:
                self.evaluator = SimpleEvaluator()
        # else: unknown component, ignore
