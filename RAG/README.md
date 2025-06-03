# RAG Modular Application

A clean, modular implementation of a RAG (Retrieval-Augmented Generation) application following clean architecture principles.

## Architecture Overview

This application follows a clean architecture approach with the following layers:

### Core Layer
- **Interfaces**: Abstract base classes defining contracts for components
- **Config**: Configuration management and settings
- **Exceptions**: Custom exception classes
- **Events**: Event bus and event handlers for loose coupling

### Domain Layer
- **Models**: Core business entities
- **Services**: Core business logic

### Application Layer
- **Use Cases**: Application-specific business rules orchestrating domain services

### Infrastructure Layer
- **LLM Clients**: Implementations for various LLM providers
- **Vector Stores**: Implementations for vector databases
- **Embedders**: Implementations for text embedding services
- **Chunkers**: Implementations for document chunking strategies

### Presentation Layer
- **Components**: Reusable UI components
- **Pages**: Complete page layouts

## Required Dependencies

To run this application, you need to install the following dependencies:

```bash
pip install streamlit openai chromadb langchain python-dotenv pandas numpy python-docx PyPDF2
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Default Settings
DEFAULT_LLM_SERVICE=openai
DEFAULT_LLM_MODEL=gpt-3.5-turbo
DEFAULT_EMBEDDER=openai
DEFAULT_VECTOR_STORE=memory
DEFAULT_CHUNKER=recursive
DEFAULT_RETRIEVER=similarity
DEFAULT_RERANKER=none

# Application Settings
APP_NAME=RAG Modular
DEBUG=true
```

## Running the Application

To run the application:

```bash
streamlit run main.py
```

## Features

- Document processing and chunking
- Vector storage and retrieval
- LLM-powered chat interface
- Flashcard generation
- Evaluation metrics
- Configurable components
- Event-driven architecture

## Extending the Application

### Adding a New LLM Provider

1. Create a new class in `app/infrastructure/llm/` that implements the `LLMService` interface
2. Register the new provider in the `create_llm` factory function in `main.py`

### Adding a New Vector Store

1. Create a new class in `app/infrastructure/vector_store/` that implements the `VectorStore` interface
2. Register the new vector store in the `create_vector_store` factory function in `main.py`

### Adding a New UI Component

1. Create a new component class in `app/presentation/components/`
2. Integrate the component in the appropriate page in `app/presentation/pages/`
