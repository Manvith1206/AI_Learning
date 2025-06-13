import os
import openai
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from .base_retriever import BaseRetriever
import infrastructure.Common.RAG_Constants as constants
import Utils.Utils as Utils
import time

class SentenceWindowRetriever(BaseRetriever):

    def __init__(self, window_size: int, top_k: int):
        self.window_size = window_size
        self.top_k = top_k
        self.documents = None
        self.document = None
        self.time_taken = 0
        self.cost = 0
        openai.api_key = Utils.get_env_var(constants.OPENAI_API_KEY)

    def get_sentence_window_index(self, documents, index_dir, sentence_window_size=3):
        Node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_sentence",
        )

        Settings.llm = OpenAI()
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
        Settings.node_parser = Node_parser

        if not os.path.exists(index_dir):
            sentence_index = VectorStoreIndex.from_documents([self.document])
            sentence_index.storage_context.persist(persist_dir=index_dir)
            
        else:
            sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_dir))
        return sentence_index

    def get_sentence_window_engine(self, sentence_index):
        
        postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
        sentence_window_engine = sentence_index.as_query_engine(similarity_top_k=self.top_k, node_postprocessors=[postprocessor])
        
        return sentence_window_engine
    
    def retrieve(self, query_embedding, documents, **kwargs):
        start_time = time.time()

        self.documents = documents
        print("Docs: ", self.documents)
        self.document = Document(text="\n\n".join([doc["page_content"] for doc in self.documents]))

        index_dir = "./sentence_index_1"
        print("WindowSize", self.window_size)
        sw_index_1 = self.get_sentence_window_index(documents, index_dir, sentence_window_size=self.window_size)
        sw_engine_1 = self.get_sentence_window_engine(sw_index_1)

        query_text = kwargs.get("query_text")
        window_response_1 = sw_engine_1.query(
            query_text
        )
        retrieved_contexts = [sn.node.metadata["window"] for sn in window_response_1.source_nodes]
        end_time = time.time()
        self.time_taken = end_time - start_time
        print("Sentence WIndow / Retrieved Contexts: " )
        print(retrieved_contexts)
        return retrieved_contexts
        
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken