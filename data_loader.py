import os
from pathlib import Path
import nltk
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


nest_asyncio.apply()

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Cấu hình LLM và Embedding
Settings.llm = Ollama(model="deepseek-r1:1.5b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()

# Cấu hình chunk size
Settings.chunk_size = 512
storage_context = StorageContext.from_defaults()

# Tạo index với cài đặt mới
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
storage_context.persist(persist_dir="./storage/")