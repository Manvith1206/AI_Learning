import os
import openai
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
import infrastructure.common.rag_constants as constants
import Utils.utils as Utils
import time

class SentenceWindowRetriever(BaseRetriever):

    def __init__(self, window_size: int, top_k: int):
        self.window_size = window_size
        self.top_k = top_k
        self.documents = None
        self.document = None
        self.time_taken = 0
        self.cost = 0
        openai.api_key = Utils.get_env_var(constants.APIKeys.OPENAI_API_KEY)

    def get_sentence_window_index(self, documents, index_dir, sentence_window_size=3):
        Node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=sentence_window_size,
            window_metadata_key=constants.Constants.WINDOW_METADATA_KEY,
            original_text_metadata_key=constants.Constants.ORIGINAL_TEXT_METADATA_KEY,
        )

        Settings.llm = OpenAI()
        Settings.embed_model = constants.Constants.BGE_SMALL_EMBED_MODEL
        Settings.node_parser = Node_parser

        if not os.path.exists(index_dir):
            sentence_index = VectorStoreIndex.from_documents([self.document])
            sentence_index.storage_context.persist(persist_dir=index_dir)
            
        else:
            sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_dir))
        return sentence_index

    def get_sentence_window_engine(self, sentence_index):
        
        postprocessor = MetadataReplacementPostProcessor(target_metadata_key=constants.Constants.WINDOW_METADATA_KEY)
        sentence_window_engine = sentence_index.as_query_engine(similarity_top_k=self.top_k, node_postprocessors=[postprocessor])
        
        return sentence_window_engine
    
    def retrieve(self, query_embedding, documents, **kwargs):
        start_time = time.time()

        self.documents = documents
        self.document = Document(text="\n\n".join([doc[constants.Constants.PAGE_CONTENT] for doc in self.documents]))

        index_dir = "./sentence_index_1"
        sw_index_1 = self.get_sentence_window_index(documents, index_dir, sentence_window_size=self.window_size)
        sw_engine_1 = self.get_sentence_window_engine(sw_index_1)

        query_text = kwargs.get(constants.Constants.QUERY_TEXT)
        window_response_1 = sw_engine_1.query(
            query_text
        )
        retrieved_contexts = [sn.node.metadata[constants.Constants.WINDOW_METADATA_KEY] for sn in window_response_1.source_nodes]
        end_time = time.time()
        self.time_taken = end_time - start_time
        return retrieved_contexts
        
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken