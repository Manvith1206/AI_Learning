import warnings
warnings.filterwarnings('ignore')

import streamlit as st

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)
import numpy as np

from trulens_eval.feedback import groundtruth 

import os
import openai
os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_AI_API_KEY"]

TEMP_DIR = "temp_docs"
from dotenv import load_dotenv, find_dotenv

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())
    # To this
    return os.getenv("OPENAI_API_KEY")

openai.api_key = get_openai_api_key()

from llama_index.core import SimpleDirectoryReader

def get_prebuilt_trulens_recorder(query_engine, app_id):
    openai = OpenAI()

    qa_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )

    qs_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )

    feedbacks = [qa_relevance, qs_relevance]
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

uploaded_file = st.file_uploader("Upload a file for Processing", ["pdf", "txt"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    # Write to temp file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    print("File Path: " + file_path)
    documents = SimpleDirectoryReader(
        input_files=[file_path]
    ).load_data()

    from llama_index.core import Document

    document = Document(text="\n\n".join([doc.text for doc in documents]))

    def build_sentence_window_index(
        documents,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        sentence_window_size=3,
        save_dir="sentence_index",
    ):
        # create the sentence window node parser w/ default settings
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        if not os.path.exists(save_dir):
            sentence_index = VectorStoreIndex.from_documents(
                documents
            )
            sentence_index.storage_context.persist(persist_dir=save_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=save_dir)
            sentence_index = load_index_from_storage(storage_context)
            # sentence_index = load_index_from_storage(
            #     StorageContext.from_defaults(persist_dir=save_dir),
            #     service_context=sentence_context,
            # )

        return sentence_index


    def get_sentence_window_query_engine(
        sentence_index, similarity_top_k=6, rerank_top_n=2
    ):
        # define postprocessors
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )

        sentence_window_engine = sentence_index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
        )
        return sentence_window_engine
    
    from llama_index.llms import openai
    
    index = build_sentence_window_index(
        [document],
        llm=OpenAI(model_engine="gpt-3.5-turbo"),
        save_dir="./sentence_index",
    )

    query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)

    eval_questions = []
    with open('RAG/generated_questions.text', 'r') as file:
        for line in file:
            # Remove newline character and convert to integer
            item = line.strip()
            eval_questions.append(item)

    from trulens_eval import Tru

    def run_evals(eval_questions, tru_recorder, query_engine):
        for question in eval_questions:
            with tru_recorder as recording:
                response = query_engine.query(question)

    from trulens_eval import Tru

    Tru().reset_database()

    sentence_index_1 = build_sentence_window_index(
    documents,
    llm=OpenAI(model_engine="gpt-3.5-turbo"),
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=1,
    save_dir="sentence_index_1",
    )

    sentence_window_engine_1 = get_sentence_window_query_engine(
    sentence_index_1
    )

    tru_recorder_1 = get_prebuilt_trulens_recorder(
    sentence_window_engine_1,
    app_id='sentence window engine 1'
    )

    run_evals(eval_questions, tru_recorder_1, sentence_window_engine_1)

    Tru().run_dashboard()


    