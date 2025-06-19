import itertools
import infrastructure.Common.RAG_Constants as constants
from infrastructure.Common.RAG_Constants import (
    ChunkerType, EmbedderType, RetrieverType, RerankerType, LLMServiceType, EvaluatorType
)

# Define the parameter options for each component of the RAG pipeline.
# These lists will be used to generate all possible testing configurations.

CHUNKERS = [
    {
        constants.CONFIG_TYPE_PARAM: ChunkerType.RECURSIVE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_CHUNK_SIZE_PARAM: 150,
            constants.CONFIG_CHUNK_OVERLAP_PARAM: 70
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: ChunkerType.SENTENCE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_MAX_SENTENCES: 18
        }
    }
]

EMBEDDERS = [
    {
        constants.CONFIG_TYPE_PARAM: EmbedderType.VOYAGE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_MODEL: constants.VoyageEmbedModels.VOYAGE_3_LITE_EMBED_MODEL.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: EmbedderType.VOYAGE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_MODEL: constants.VoyageEmbedModels.VOYAGE_3_EMBED_MODEL.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: EmbedderType.VOYAGE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_MODEL: constants.VoyageEmbedModels.VOYAGE_EMBED_DEFAULT_MODEL.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: EmbedderType.COHERE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_MODEL: constants.CohereEmbedModels.COHERE_EMBED_MODEL_DEFAULT.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: EmbedderType.COHERE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_MODEL: constants.CohereEmbedModels.COHERE_EMBED_MODEL_ENG.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: EmbedderType.COHERE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_MODEL: constants.CohereEmbedModels.COHERE_EMBEDDING_MULTILINGUAL_V3_0.value
        }
    },
]

VECTOR_STORES = [
    {
        constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_FAISS,
        constants.CONFIG_PARAM: {}
    },
    {
        constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_PINCONE,
        constants.CONFIG_PARAM: {}
    }
]

RETRIEVERS = [
    {
        constants.CONFIG_TYPE_PARAM: RetrieverType.SIMILARITY.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0,
            constants.CONFIG_TOP_K_PARAM: 20
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: RetrieverType.HYBRID.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_KEYWORD_WEIGHT: 0.5,
            constants.CONFIG_TOP_K_PARAM: 20
        }
    }
]

RERANKERS = [
    {
        constants.CONFIG_TYPE_PARAM: RerankerType.COHERE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5,
            constants.CONFIG_MODEL: constants.CohereRerankingModels.RERANK_DEFAULT_MODEL.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: RerankerType.COHERE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5,
            constants.CONFIG_MODEL: constants.CohereRerankingModels.RERANK_ENGLISH.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: RerankerType.COHERE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5,
            constants.CONFIG_MODEL: constants.CohereRerankingModels.RERANK_MULTLINGUAL.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: RerankerType.LLM.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: RerankerType.JINA.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5,
            constants.CONFIG_MODEL: constants.JINA_RERANKER_MODELS.JINA_RERANKER_MULTILINGUAL.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: RerankerType.JINA.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5,
            constants.CONFIG_MODEL: constants.JINA_RERANKER_MODELS.JINA_RERANKER_V1_TURBO.value
        }
    }
]

LLMS = [
    {
        constants.CONFIG_TYPE_PARAM: LLMServiceType.GEMINI.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_MODEL: constants.GeminiLLMModel.GEMINI_PRO.value
        }
    },
    {
        constants.CONFIG_TYPE_PARAM: LLMServiceType.CLAUDE.value,
        constants.CONFIG_PARAM: {
            constants.CONFIG_MODEL: constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_7.value
        }
    }
]

EVALUATORS = [
    {
        constants.CONFIG_TYPE_PARAM: EvaluatorType.RAGAS.value
    }
]

def generate_configurations():
    """Generates all possible combinations of RAG components."""
    product = itertools.product(
        CHUNKERS, EMBEDDERS, VECTOR_STORES, RETRIEVERS, RERANKERS, LLMS, EVALUATORS
    )

    configs = []
    for (chunker, embedder, vs, retriever, reranker, llm, evaluator) in product:
        config = {
            constants.CONFIG_CHUNKER: chunker,
            constants.CONFIG_EMBEDDER: embedder,
            constants.CONFIG_VECTOR_STORE: vs,
            constants.CONFIG_RETRIEVER: retriever,
            constants.CONFIG_RERANKER: reranker,
            constants.CONFIG_LLM: llm,
            constants.CONFIG_EVALUATOR: evaluator
        }
        configs.append(config)
    
    import json
    with open("Configs.txt", "w", encoding="utf-8") as file:
        file.write(json.dumps(configs, indent=2))

    return configs

if __name__ == '__main__':
    # Example of how to generate and print the configurations
    all_configs = generate_configurations()
    print(f"Generated {len(all_configs)} configurations.")
    # You can uncomment the line below to see all the generated configurations
    
