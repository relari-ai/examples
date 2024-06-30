from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


def preprocess_documents(doc_dir: Path):
    document_store = InMemoryDocumentStore()
    text_file_converter = TextFileToDocument()
    document_cleaner = DocumentCleaner()
    document_joiner = DocumentJoiner()
    document_splitter = DocumentSplitter(split_by="sentence", split_length=2)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_writer = DocumentWriter(document_store)

    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(
        instance=text_file_converter, name="text_file_converter"
    )
    preprocessing_pipeline.add_component(
        instance=document_joiner, name="document_joiner"
    )
    preprocessing_pipeline.add_component(
        instance=document_cleaner, name="document_cleaner"
    )
    preprocessing_pipeline.add_component(
        instance=document_splitter, name="document_splitter"
    )
    preprocessing_pipeline.add_component(
        instance=document_embedder, name="document_embedder"
    )
    preprocessing_pipeline.add_component(
        instance=document_writer, name="document_writer"
    )
    preprocessing_pipeline.connect("text_file_converter", "document_joiner")
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    sources = [
        str(f) for f in doc_dir.iterdir() if f.is_file() and f.suffix in [".txt"]
    ]
    preprocessing_pipeline.run({"text_file_converter": {"sources": sources}})
    return document_store
