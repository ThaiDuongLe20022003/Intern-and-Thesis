import os
import chainlit as cl
from langchain_core.documents import Document as LCDocument
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableConfig
from langchain.schema import StrOutputParser
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
from operator import itemgetter

# Load environment variables
load_dotenv()

# Model Settings
ollama_model = "deepseek-llm:latest"
qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
collection_name = "deeplaw"

# Define custom prompt template with history
custom_prompt_template = """You are a professional legal expert. Always cite based the Law on Cyberinformation Security 2015 and Law on Cybersecurity 2018; and answer concisely.

Use the following pieces of information and conversation history (if any) to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Conversation History (if any):
{history}

Question: {question}

Helpful answer:
"""

def format_docs(docs: list[LCDocument]) -> str:
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(messages: list[ChatMessage]) -> str:
    """Format chat history for context"""
    history_str = ""
    for msg in messages:
        if msg.role == "user":
            history_str += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            history_str += f"Assistant: {msg.content}\n"
    return history_str.strip()

@cl.on_chat_start
async def start():
    # Initialize Qdrant client and vector store
    client = QdrantClient(url=qdrant_url)
    embeddings = FastEmbedEmbeddings()
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    # Initialize LLM
    llm = OllamaLLM(model=ollama_model, temperature=0)
    
    # Create RAG chain (will be completed in on_message)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("llm", llm)
    
    # Set up chat history
    CHAT_FILE_PATH = "./chat/chat_store.json"
    chat_store = SimpleChatStore()
    
    # Create chat directory if it doesn't exist
    os.makedirs(os.path.dirname(CHAT_FILE_PATH), exist_ok=True)
    
    # Load existing chat history if available
    if os.path.exists(CHAT_FILE_PATH) and os.path.getsize(CHAT_FILE_PATH) > 0:
        try:
            chat_store = SimpleChatStore.from_persist_path(CHAT_FILE_PATH)
        except:
            # Create new store if loading fails
            chat_store = SimpleChatStore()

    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=1500,
        chat_store=chat_store,
        chat_store_key="user"
    )
    
    # Store objects in user session
    cl.user_session.set("chat_memory", chat_memory)
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
    # Retrieve objects from session
    retriever = cl.user_session.get("retriever")
    llm = cl.user_session.get("llm")
    chat_memory = cl.user_session.get("chat_memory")
    chat_store = cl.user_session.get("chat_store")
    
    # Get conversation history
    history_messages = chat_memory.get()
    history_str = format_history(history_messages)
    
    # Create RAG chain with history
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    runnable = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "history": itemgetter("history"),
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Add user message to memory
    chat_memory.put(ChatMessage(role="user", content=message.content))
    
    # Generate response
    msg = cl.Message(content="")
    response = ""
    
    # Create input with history
    inputs = {
        "question": message.content,
        "history": history_str
    }
    
    async for chunk in runnable.astream(
        inputs,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    ):
        await msg.stream_token(chunk)
        response += chunk
    
    # Add assistant response to memory
    chat_memory.put(ChatMessage(role="assistant", content=response))
    
    # Finalize and persist
    await msg.send()
    chat_store.persist(persist_path="./chat/chat_store.json")

if __name__ == "__main__":
    cl.run(main)