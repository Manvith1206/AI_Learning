import uuid
import traceback
import infrastructure.common.rag_constants as constants
import os
from infrastructure.chunkers.base_chunker import BaseChunker
from infrastructure.embedders.base_embedder import BaseEmbedder
from infrastructure.vector_stores.base_vector_store import BaseVectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

class DocumentProcessing:
    def __init__(self, error_callback, process_doc_callback, vector_store: BaseVectorStore, embedder: BaseEmbedder, chunker: BaseChunker):
        self.error_callback = error_callback
        self.process_doc_callback = process_doc_callback
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker

    def extractText(self, file: UploadedFile, temp_dir=constants.Constants.TEMP_DOCS_DIR):
        try:
            from infrastructure.document_loaders.pdf_loader import PDFLoader
            from infrastructure.document_loaders.docx_loader import DOCXLoader
            from infrastructure.document_loaders.txt_loader import TXTLoader
            from infrastructure.document_loaders.csv_loader import CSVLoader
            loaders = {
                constants.FileExtensionConstants.PDF_EXTENSION: PDFLoader(),
                constants.FileExtensionConstants.DOCX_EXTENSION: DOCXLoader(),
                constants.FileExtensionConstants.TXT_EXTENSION: TXTLoader(),
                constants.FileExtensionConstants.CSV_EXTENSION: CSVLoader(),
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
            with open(constants.Constants.EXTRACTED_TEXT_FILE_PATH, "w", encoding="utf-8") as file:
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
                    constants.Constants.ID: doc_id,
                    constants.Constants.PAGE_CONTENT: chunk,
                    constants.Constants.METADATA: {"source": file.name}
                })
            texts = [doc[constants.Constants.PAGE_CONTENT] for doc in documents]
            
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
        