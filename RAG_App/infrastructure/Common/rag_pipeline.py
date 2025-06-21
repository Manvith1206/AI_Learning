import os
from typing import Dict, List
import uuid

from infrastructure.evaluators.simple_evaluator  import SimpleEvaluator
from infrastructure.evaluators.ragas_evaluator import RagasEvaluator

from config import ConfigManager

import infrastructure.common.rag_constants as constants
from infrastructure.common.query_classifier_llm import QueryClassifier

import traceback
import json # Added for parsing LLM response for flashcards
from infrastructure.evaluators.deep_eval_evaluator import DeepEval
import Utils.exceptions as Exceptions
from infrastructure.prompt_providers.llm_chat_prompt_provider import LLM_Chat_Prompt_Provider
from infrastructure.prompt_providers.flashcards_generation_prompt_provider import FlashCardsGeneration_Prompt_Provider

class RAGPipeline:
    def __init__(self, 
                geminiApiKey,
                cohereApiKey, 
                voyageApiKey, 
                mistralApiKey, 
                pineconeApiKey, 
                jinaApiKey, 
                claudeApiKey, 
                warning_callback, 
                error_callback, 
                process_doc_callback, 
                config_manager=None, 
                vector_store=None):
        
        self.config_manager = config_manager or ConfigManager()
        self.vector_store = vector_store
        self.warning_callback = warning_callback
        self.error_callback = error_callback
        self.process_doc_callback = process_doc_callback
        
        # API Keys assignment (if needed for further use within the pipeline)
        self.geminiApiKey = geminiApiKey
        self.cohereApiKey = cohereApiKey
        self.voyageApiKey = voyageApiKey
        self.mistralApiKey = mistralApiKey
        self.pineconeApiKey = pineconeApiKey
        self.jinaApiKey = jinaApiKey
        self.claudeApiKey = claudeApiKey
        
        self.query_classifier = None
        self.flashcard_prompt_provider = FlashCardsGeneration_Prompt_Provider()

class DocumentProcessing:
    def __init__(self, error_callback):
        self.error_callback = error_callback
    def extractText(self, file, temp_dir=constants.TEMP_DOCS_DIR):
        try:
            from infrastructure.document_loaders.pdf_loader import PDFLoader
            from infrastructure.document_loaders.docx_loader import DOCXLoader
            from infrastructure.document_loaders.txt_loader import TXTLoader
            from infrastructure.document_loaders.csv_loader import CSVLoader
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
            if file_ext in loaders:
                text = loaders[file_ext].load_document(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # # Remove headers and footers
            # text = re.sub(r"DCA2104: Basics of Data Communication Manipal University Jaipur$MUJ$", "", text)

            # # Remove page numbers or line numbers
            # text = re.sub(r"Unit \d+:.*", "", text)
            # text = re.sub(r"\d+\s*$", "", text)

            # # Remove extra whitespaces and newlines
            # text = re.sub(r"\s+", " ", text).strip()
            with open("ExtractedTextFromPdf.txt", "w", encoding="utf-8") as file:
                file.write(text)

            if not text:
                return None, None
            else:
                return text
        except Exception as e:
            self.error_callback(f"Error extracting text: {e}, Traceback: {traceback.print_exc()}")
            return None, None
         
    # process documents
    def process_document(self, file, texts=None):
        try:
            chunks =  self.chunker.split_text(text=texts)

            
            documents = []
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                documents.append({
                    constants.ID: doc_id,
                    constants.PAGE_CONTENT: chunk,
                    constants.METADATA: {"source": file.name}
                })
            texts = [doc[constants.PAGE_CONTENT] for doc in documents]
            
            embeddings = self.embedder.embed_documents(texts)
            
            
            documents = self.vector_store.format_documents(documents)
            self.vector_store.add_embeddings(embeddings, documents)
            self.vector_store.documents = documents  # Attach documents for caching
            self.process_doc_callback(f"Document Processed Succesfully with Chunks: {len(chunks)}")
            
            return self.vector_store
        except Exception as e:
            full_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.error_callback(f"Error processing document: {e}, Traceback: {full_traceback}")
            return None
        
class QueryProcessing:
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
        
    def get_context_docs(self, query_text, top_k=None):
        
        if not hasattr(self.vector_store, 'documents') or not self.vector_store.documents:
                raise ValueError("No documents processed. Please upload and process a document before querying.")
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
        reranked_docs, explanation = self.reranker.rerank(query_text, retrieved_docs, top_k=top_k)
        
        
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
        
class QueryEvaluation:
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
        try:
            # Use last query data if not provided
            
            if hasattr(self, constants.LAST_QUERY) and (question is None or answer is None or contexts is None):
                question = question or self.last_query[constants.QUESTION]
                answer = answer or self.last_query[constants.ANSWER]
                contexts = contexts or self.last_query[constants.CONTEXTS]
            
            if not (question and answer and contexts):
                raise ValueError("No query data available for evaluation")
            
            # Run evaluation
            metrics = self.evaluator.evaluate(question, answer, contexts, ground_truths)
            return metrics
        except Exception as e:
            raise Exceptions.EvaluationError("Error During Evaluation")

class FlashCardGeneration:
    def generate_flashcards_from_text(self, text_content: str, num_flashcards: int = 5) -> List[Dict[str, str]]:
        """Generates flashcards from the given text content using the LLM service."""
        if not text_content.strip():
            self.warning_callback("Cannot generate flashcards from empty content.")
            return []

        prompt = self.flashcard_prompt_provider.get_final_prompt(text_content=text_content, num_flashcards=num_flashcards)
        
        try:
            full_response = ""
            # Assuming llm_service.generate_response is a generator yielding response chunks
            for delta in self.llm_service.generate_response(prompt):
                full_response += delta
            
            # Attempt to parse the LLM's response as JSON
            # The response might be wrapped in markdown code blocks, try to strip them
            if full_response.strip().startswith("```json"):
                full_response = full_response.strip()[7:-3].strip()
            elif full_response.strip().startswith("```"):
                 full_response = full_response.strip()[3:-3].strip()

            flashcards = json.loads(full_response)
            
            # Validate structure
            if not isinstance(flashcards, list):
                raise ValueError("LLM response is not a list.")
            for card in flashcards:
                if not (isinstance(card, dict) and "question" in card and "answer" in card):
                    raise ValueError("Invalid flashcard structure in LLM response.")
            
            return flashcards[:num_flashcards] # Return up to the requested number

        except json.JSONDecodeError as e:
            self.error_callback(f"Error decoding JSON from LLM for flashcards: {e}\nRaw response: {full_response}")
            print(f"JSONDecodeError: {e}. Raw LLM response for flashcards:\n{full_response}")
            return []
        except ValueError as e:
            self.error_callback(f"Error in flashcard data structure from LLM: {e}\nRaw response: {full_response}")
            print(f"ValueError: {e}. Raw LLM response for flashcards:\n{full_response}")
            return []
        except Exception as e:
            self.error_callback(f"An unexpected error occurred during flashcard generation: {e}")
            print(f"Unexpected error in generate_flashcards_from_text: {e}, Traceback: {traceback.format_exc()}")
            return []
