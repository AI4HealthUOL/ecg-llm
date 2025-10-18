import glob
import os
import sys
from typing import List

import chromadb
import tiktoken
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()
import glob
import os
from typing import List

import chromadb
import tiktoken
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import SentenceTransformer

# https://medium.com/@praveencs87/chunking-in-rag-retrieval-augmented-generation-a-beginners-guide-28b5a81a8877 markdown splitting, weil markdown dokumente
# https://docs.trychroma.com/docs/embeddings/embedding-functions

# https://research.trychroma.com/evaluating-chunking
# https://huggingface.co/NeuML/pubmedbert-base-embeddings


# https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/
class PubMedEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input[0], str):
            return self.model.encode(input)
        return self.model.encode([doc.page_content for doc in input])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.encode(text)


def token_length(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def create_chroma_collection_pubmedBert_recursive_splitting(DB_PATH, FOLDER_PATH, collection_name):

    logger.add(
        "/logs/recursive_splitting_ecg.log", format="{time} {level} {message}", level="INFO"
    )
    client = chromadb.PersistentClient(path=DB_PATH)

    # https://huggingface.co/NeuML/pubmedbert-base-embeddings
    logger.info(collection_name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    logger.info("Model loaded")
    collections = client.list_collections()
    print(collections)
    documents = []

    for file_path in glob.glob(os.path.join(FOLDER_PATH, "*.md")):
        try:

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    if not chunk.strip():
                        logger.info(f"Skipping empty chunk in {file_path}")
                        continue
                    doc = Document(page_content=chunk, metadata={"source": file_path})
                    documents.append(doc)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    Chroma.from_documents(
        documents=documents,
        embedding=PubMedEmbedding(),
        collection_name=collection_name,
        client=client,
    )
    logger.info(f"Added {len(chunks)} chunks to chroma db")


if __name__ == "__main__":
    FOLDER_PATH = "/output_cleaned_markdown_pipeline"
    DB_PATH = "/chroma_db_master_thesis_pubmedbert_recursive_reranking"
    logger.add(
        "/logs/markdown_splitting_ecg.log", format="{time} {level} {message}", level="INFO"
    )
    create_chroma_collection_pubmedBert_recursive_splitting(DB_PATH, FOLDER_PATH, "ecg_haverkamps_markdowns")
