from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import os

class PDFProcessor:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model)
        self.node_parser = SimpleNodeParser.from_defaults()
        self.vector_store = None
        self.index = None
        
        # Set the embedding model globally
        Settings.embed_model = self.embed_model

    def load_and_process_pdf(self, pdf_path: str, index_name: str = "pdf_index"):
        # Load PDF
        loader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = loader.load_data()

        # Split into nodes
        nodes = self.node_parser.get_nodes_from_documents(documents)

        # Create vector store
        self.vector_store = SimpleVectorStore()
        
        # Create storage context with the vector store
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create index
        self.index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=self.embed_model
        )

        # Persist index
        self.persist_index(index_name)

        return self.index

    def persist_index(self, index_name: str):
        # Create directory if it doesn't exist
        os.makedirs(index_name, exist_ok=True)

        # Save the index
        self.index.storage_context.persist(persist_dir=index_name)

    def load_index(self, index_name: str):
        # Load the index
        storage_context = StorageContext.from_defaults(persist_dir=index_name)
        self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        return self.index

    def query(self, query_text: str) -> str:
        if self.index is None:
            raise ValueError("Index not created or loaded. Please process a PDF or load an existing index first.")
        
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        return str(response)

