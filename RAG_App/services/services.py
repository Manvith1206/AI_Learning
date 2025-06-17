from infrastructure.Common.rag_pipeline import RAGPipeline

class DocumentProcessor:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        
    def process_uploaded_file(self, uploaded_file) -> tuple:
        """Process an uploaded document and return documents and chunks"""
        texts = self.pipeline.extractText(uploaded_file)
        return self.pipeline.process_document(uploaded_file, texts)