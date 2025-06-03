from typing import Dict, List, Any, Optional, Tuple
from RAG_App.infrastructure.Common.rag_pipeline import RAGPipeline
from RAG_App.models import Document, DocumentChunk
from RAG_App.infrastructure.Chunkers import base_chunker as Chunker
from RAG_App.infrastructure.Embedders import base_embedder as Embedder
from RAG_App.infrastructure.Vector_Stores import base_vector_store as VectorStore
from RAG_App.infrastructure.Retrieval_Methods import base_retriever as Embedder
from RAG_App.infrastructure.Rerankers import base_reranker as Reranker
from RAG_App.infrastructure.LLM_Chat_Services import base_llm_service as LLMService
from RAG_App.infrastructure.Evaluators import base_evaluator as Evaluator
from RAG_App.models import QueryResult, Flashcard, EvaluationResult

class DocumentProcessor:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        
    def process_uploaded_file(self, uploaded_file) -> tuple:
        """Process an uploaded document and return documents and chunks"""
        texts = self.pipeline.extractText(uploaded_file)
        return self.pipeline.process_document(uploaded_file, texts)
    
class RAGService:
    """Core domain service for RAG operations"""
    
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        retriever: Embedder,
        reranker: Reranker,
        llm_service: LLMService,
        evaluator: Evaluator
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker = reranker
        self.llm_service = llm_service
        self.evaluator = evaluator
        self.last_query: Optional[str] = None
        self.last_result: Optional[QueryResult] = None
    
    def process_document(self, document: Document) -> Document:
        """Process a document by chunking and embedding it"""
        # Chunk the document
        chunks = self.chunker.chunk_text(document.content)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                id=f"{document.id}_chunk_{i}",
                content=chunk_text,
                document_id=document.id,
                metadata=document.metadata.copy()
            )
            document_chunks.append(chunk)
            document.add_chunk(chunk)
        
        # Embed the chunks
        chunk_texts = [chunk.content for chunk in document_chunks]
        embeddings = self.embedder.embed_documents(chunk_texts)
        
        # Assign embeddings to chunks
        for chunk, embedding in zip(document_chunks, embeddings):
            chunk.embedding = embedding
        
        # Store in vector store
        self.vector_store.store_embeddings(
            [chunk.__dict__ for chunk in document_chunks],
            embeddings
        )
        
        return document
    
    def query(self, query: str) -> QueryResult:
        """Execute a RAG query and return results"""
        # Store the query for later evaluation
        self.last_query = query
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query)
        
        # Rerank documents
        reranked_docs, rerank_explanation = self.reranker.rerank(query, retrieved_docs)
        
        # Generate answer using LLM
        context = "\n\n".join([doc.get("content", "") for doc in reranked_docs])
        prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        llm_response = self.llm_service.query(prompt)
        answer = llm_response.get("answer", "")
        
        # Create and store result
        result = QueryResult(
            query=query,
            answer=answer,
            retrieved_documents=[
                DocumentChunk(
                    id=doc.get("id", ""),
                    content=doc.get("content", ""),
                    document_id=doc.get("document_id", ""),
                    metadata=doc.get("metadata", {})
                )
                for doc in reranked_docs
            ],
            rerank_explanation=rerank_explanation,
            metadata={"raw_llm_response": llm_response}
        )
        
        self.last_result = result
        return result
    
    def evaluate(self, ground_truth: str) -> EvaluationResult:
        """Evaluate the last query against a ground truth"""
        if not self.last_query or not self.last_result:
            raise ValueError("No query has been executed yet")
        
        metrics = self.evaluator.evaluate(
            query=self.last_query,
            retrieved_docs=[chunk.__dict__ for chunk in self.last_result.retrieved_documents],
            answer=self.last_result.answer,
            ground_truth=ground_truth
        )
        
        return EvaluationResult(
            metrics=metrics,
            query=self.last_query,
            answer=self.last_result.answer,
            ground_truth=ground_truth,
            retrieved_documents=self.last_result.retrieved_documents
        )
    
    def generate_flashcards(self, document: Document, num_cards: int = 5) -> List[Flashcard]:
        """Generate flashcards from a document"""
        flashcards = []
        
        # Use chunks to generate flashcards
        for chunk in document.chunks[:num_cards]:  # Limit to requested number of cards
            prompt = f"""
            Generate a flashcard (question and answer pair) based on the following text:
            
            {chunk.content}
            
            Format:
            Question: [question]
            Answer: [answer]
            """
            
            response = self.llm_service.query(prompt)
            raw_text = response.get("answer", "")
            
            # Parse question and answer
            question = ""
            answer = ""
            
            if "Question:" in raw_text and "Answer:" in raw_text:
                parts = raw_text.split("Answer:")
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].strip()
            
            if question and answer:
                flashcard = Flashcard(
                    id=f"flashcard_{len(flashcards)}_{chunk.id}",
                    question=question,
                    answer=answer,
                    document_id=document.id,
                    document_chunk_id=chunk.id
                )
                flashcards.append(flashcard)
        
        return flashcards
    
    def get_component_metrics(self) -> Dict[str, Tuple[float, str]]:
        """Get cost and time metrics for all components"""
        return {
            "chunker": self.chunker.get_cost_and_time(),
            "embedder": self.embedder.get_cost_and_time(),
            "vector_store": self.vector_store.get_cost_and_time(),
            "retriever": self.retriever.get_cost_and_time(),
            "reranker": self.reranker.get_cost_and_time(),
            "llm_service": self.llm_service.get_cost_and_time(),
            "evaluator": self.evaluator.get_cost_and_time()
        }
