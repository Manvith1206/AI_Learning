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
from .cohere_embedder import CohereEmbedder
from rag_modular.RAG_Constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType, GeminiLLMModel
)
import rag_modular.RAG_Constants as constants
from rag_modular.similarity_retriever import SimilarityRetriever
from .sentence_window_retreiver import SentenceWindowRetriever
from .cohere_re_ranker import CohereReranker
from .voyage_embedder import VoyageEmbedder
from .FAISS_Vector_Store import FAISS_Vector_Store
from .jina_reranker import JinaReranker

class RAGPipeline:
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or ConfigManager()
        self.setup_components()

    # setup components
    def setup_components(self):
        # Build all core components via factory methods
        self.chunker = self._build_chunker()
        self.embedder = self._build_embedder()
        self.vector_store = self._build_vector_store()
        self.retriever = self._build_retriever()
        self.llm_service = self._build_llm_service()
        self.reranker = self._build_reranker()
        self.evaluator = self._build_evaluator()

    # build invidual components
    def _build_chunker(self):
        cfg = self.config_manager.get_config(constants.CONFIG_CHUNKER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        
        params = cfg.get(constants.CONFIG_PARAM, {})
        if t == ChunkerType.RECURSIVE.value:
            return RecursiveChunker(**params)
        elif t == ChunkerType.SENTENCE.value:
            return SentenceChunker(**params)
        elif t == ChunkerType.SEMANTIC.value:
            return SemanticChunker(**params)
        else:
            return RecursiveChunker()

    def _build_embedder(self):
        cfg = self.config_manager.get_config(constants.CONFIG_EMBEDDER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        model_name = cfg.get(constants.CONFIG_MODEL)

        if t == EmbedderType.TFIDF.value:
            return TFIDFEmbedder()
        elif t == EmbedderType.GEMINI.value:
            return GeminiEmbedder(api_key=st.secrets[constants.GEMINI_API_KEY], model_name=model_name)
        elif t == EmbedderType.COHERE.value:
            return CohereEmbedder(api_key=st.secrets[constants.COHERE_API_KEY],
                                  model=cfg.get(constants.CONFIG_MODEL))
        elif t == EmbedderType.VOYAGE.value:
            return VoyageEmbedder(api_key=st.secrets[constants.VOYAGE_API_KEY],
                                  model=cfg.get(constants.CONFIG_MODEL))
        else:
            return TFIDFEmbedder()

    def _build_vector_store(self):
        cfg = self.config_manager.get_config(constants.CONFIG_VECTOR_STORE)
        params = cfg.get(constants.CONFIG_PARAM, {})
        type = cfg.get(constants.CONFIG_TYPE_PARAM)
        
        if type == constants.CONFIG_VECTOR_STORE_SKLEARN:
            return SklearnVectorStore(**params)
        else:
            return FAISS_Vector_Store()

    def _build_retriever(self):
        cfg = self.config_manager.get_config(constants.CONFIG_RETRIEVER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        params = cfg.get(constants.CONFIG_PARAM, {})
        self.top_k = cfg.get(constants.CONFIG_TOP_K_PARAM, getattr(self, 'top_k', 5))
        if t == RetrieverType.SIMILARITY.value:
            return SimilarityRetriever(**params)
        elif t == RetrieverType.HYBRID.value:
            return HybridRetriever(**params)
        elif t == RetrieverType.SENTENCE_WINDOW.value:
            return SentenceWindowRetriever(**params)
        else:
            return SimilarityRetriever()

    def _build_llm_service(self):
        cfg = self.config_manager.get_config(constants.CONFIG_LLM)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        from google import genai
        client = genai.Client(api_key=st.secrets[constants.GEMINI_API_KEY])
        if t == LLMServiceType.GEMINI.value:
            return GeminiService(client, model_name=cfg.get(constants.CONFIG_MODEL))
        else:
            return GeminiService(client)

    def _build_reranker(self):
        cfg = self.config_manager.get_config(constants.CONFIG_RERANKER)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        model = cfg.get(constants.CONFIG_PARAM)
        if t == RerankerType.LLM.value:
            return LLMReranker(self.llm_service.client, model_name=model)
        elif t == RerankerType.COHERE.value:
            return CohereReranker(st.secrets[constants.COHERE_API_KEY], model_name=model)
        elif t == RerankerType.JINA.value:
            return JinaReranker(model_name=model)
        elif t == RerankerType.COSINE.value:
            return CosineReranker(self.embedder)
        else:
            return CosineReranker(self.embedder)

    def _build_evaluator(self):
        cfg = self.config_manager.get_config(constants.CONFIG_EVALUATOR)
        t = cfg.get(constants.CONFIG_TYPE_PARAM)
        if t == EvaluatorType.RAGAS.value:
            return RagasEvaluator(**cfg.get(constants.CONFIG_PARAM, {}))
        else:
            return SimpleEvaluator()

    # update components
    def update_component(self, component_name, config):
        self.config_manager.update_config(component_name, config)
        if component_name in [
            constants.CONFIG_CHUNKER,
            constants.CONFIG_EMBEDDER,
            constants.CONFIG_VECTOR_STORE,
            constants.CONFIG_LLM,
            constants.CONFIG_RERANKER
        ]:
            # heavy components: rebuild whole pipeline
            self.setup_components()
        elif component_name == constants.CONFIG_RETRIEVER:
            # hot-swap retriever only
            self.retriever = self._build_retriever()
        elif component_name == constants.CONFIG_EVALUATOR:
            # hot-swap evaluator only
            self.evaluator = self._build_evaluator()
        # else: unknown component, ignore

    # process documents
    def process_document(self, file, temp_dir=constants.TEMP_DOCS_DIR):
        
        from .document_loaders.pdf_loader import PDFLoader
        from .document_loaders.docx_loader import DOCXLoader
        from .document_loaders.txt_loader import TXTLoader
        from .document_loaders.csv_loader import CSVLoader
        loaders = {
            constants.PDF_EXTENSION: PDFLoader(),
            constants.DOCX_EXTENSION: DOCXLoader(),
            constants.TXT_EXTENSION: TXTLoader(),
            constants.CSV_EXTENSION: CSVLoader(),
        }
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
        documents = []
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            documents.append({
                constants.ID: doc_id,
                constants.PAGE_CONTENT: chunk,
                constants.METADATA: {"source": file.name}
            })
        print("Docs loaded from chunks data")
        texts = [doc[constants.PAGE_CONTENT] for doc in documents]
        print("Texts extracted successfully")
        embeddings = self.embedder.fit(texts)
        st.success(f"Embedder: {self.embedder}")
        
        print("Embeddings generated successfully")
        print(type(embeddings))
        print(type(documents))
        
        documents = self.vector_store.format_documents(documents)
        self.vector_store.add_embeddings(embeddings, documents)
        

        print("Self docs: ")
        print(self.vector_store.documents[0])
        print("Documents processed successfully")

        return documents, chunks

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
        if isinstance(query_embedding, list) and query_embedding:
            first = query_embedding[0]
            if hasattr(first, "values"):
                query_embedding = [e.values for e in query_embedding]
            elif hasattr(first, "embedding"):
                query_embedding = [e.embedding for e in query_embedding]

        print("Query embedding generated successfully")
        
        retriever_config = self.config_manager.get_config(constants.CONFIG_RETRIEVER)
        if retriever_config[constants.CONFIG_TYPE_PARAM] == constants.RetrieverType.SENTENCE_WINDOW.value:
            
            # First Retreive top docs with similarity retreival
            similarity_retreival = SimilarityRetriever(similarity_threshold=0.0)
            retreive_results = similarity_retreival.retrieve(
                query_embedding, 
                self.vector_store.documents,
                top_k=top_k,
                vector_store=self.vector_store
                )
            # After similarity retrieval:
            chunk_texts = [res[constants.Document][constants.PAGE_CONTENT] for res in retreive_results]
            # Apply the configured retriever on the filtered similarity results
            results = self.retriever.retrieve(
                query_embedding,
                chunk_texts,
                top_k=top_k,
                embedder=self.embedder,
                vector_store=self.vector_store,
                query_text=query_text
            )
            retrieved_docs = [result for result in results]
        else:
            results = self.retriever.retrieve(
                query_embedding, 
                self.vector_store.documents, 
                top_k=top_k,
                vector_store=self.vector_store,
                query_text=query_text
        )
            retrieved_docs = [result[constants.Document][constants.PAGE_CONTENT] for result in results]
            
        # Use retriever to get relevant documents
        
        print("Results retrieved successfully")
        
        # Extract document content
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
            constants.QUESTION: query_text,
            constants.ANSWER: answer,
            constants.CONTEXTS: retrieved_docs
        }
        
        
        return {
            constants.ANSWER: answer,
            constants.CONTEXTS: context_docs,
            constants.RERANK_EXPLANATION: explanation
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
        if hasattr(self, constants.LAST_QUERY) and (question is None or answer is None or contexts is None):
            question = question or self.last_query[constants.QUESTION]
            answer = answer or self.last_query[constants.ANSWER]
            contexts = contexts or self.last_query[constants.CONTEXTS]

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
