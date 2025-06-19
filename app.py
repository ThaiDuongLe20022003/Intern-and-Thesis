import os
import chainlit as cl
from llama_index.core import StorageContext, Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.storage.chat_store import SimpleChatStore
# Sửa import này - sử dụng ReActAgent từ core
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

STORAGE_DIR = "./storage"
CHAT_FILE_PATH = "./chat/chat_store.json"
DATA_DIR = "./data"

# Model Setting
Settings.llm = Ollama(model = "deepseek-r1:1.5b", temperature = 0.1, request_timeout = 120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5")

def ensure_index():
    if not os.path.exists(STORAGE_DIR) or not os.listdir(STORAGE_DIR):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir = STORAGE_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir = STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    return index

@cl.on_chat_start
async def start():
    index = ensure_index()

    if os.path.exists(CHAT_FILE_PATH) and os.path.getsize(CHAT_FILE_PATH) > 0:
        try:
            chat_store = SimpleChatStore.from_persist_path(CHAT_FILE_PATH)
        except:
            chat_store = SimpleChatStore()
    else:
        chat_store = SimpleChatStore()

    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit = 1500,
        chat_store = chat_store,
        chat_store_key = "user"
    )

    individual_query_engine_tools = [
        QueryEngineTool(
            query_engine=index.as_query_engine(),
            metadata=ToolMetadata(
                name = "law", 
                description = "Useful for answering queries about the Law on Cyberinformation Security 2015 and" \
                " Law on Cybersecurity 2018."
            )
        )
    ]

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools = individual_query_engine_tools,
        llm = Settings.llm,
    )

    tools = individual_query_engine_tools + [
        QueryEngineTool(
            query_engine = query_engine,
            metadata = ToolMetadata(
                name = "sub_question_query_engine",
                description = "Useful for answering detailed queries about the Law on Cyberinformation Security 2015 and" \
                " Law on Cybersecurity 2018."
            )
        )
    ]

    # ReActAgent from core
    agent = ReActAgent.from_tools(
        tools, 
        llm = Settings.llm,
        memory = chat_memory,
        verbose = True,
        system_prompt = "You are a professional legal expert. Always cite based the Law on Cyberinformation Security 2015 and" \
        " Law on Cybersecurity 2018; and answer concisely."
    )
    
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_store", chat_store)

@cl.on_chat_resume
async def resume():
    await start()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    users = {"Duong": "10421012"}
    if username in users and users[username] == password:
        return cl.User(identifier=username, metadata={"role": username, "provider": "credentials"})
    return None

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    chat_store = cl.user_session.get("chat_store")

    msg = cl.Message(content="", author="Assistant")
    response = await agent.astream_chat(message.content)
    
    async for token in response.async_response_gen():
        await msg.stream_token(token)
        
    await msg.send()
    chat_store.persist(persist_path = CHAT_FILE_PATH)

if __name__ == "__main__":
    cl.run(main)