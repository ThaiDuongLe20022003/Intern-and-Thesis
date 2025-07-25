import os
import nest_asyncio  
nest_asyncio.apply()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# bring in our QDRANT_URL_LOCALHOST
from dotenv import load_dotenv
load_dotenv()


qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
chunk_size = 1200
chunk_overlap = 200


import pickle
# Define a function to load parsed data if available, or parse if not
def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"
    
    if os.path.exists(data_file):
        # Load the parsed data from the file
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown")
        file_extractor = {".pdf": parser}
        llama_parse_documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
        

        # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents
    
    return parsed_data
            

# Create vector database
def create_vector_database():
    
     # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data()

    with open('data/output.md', 'w', encoding='utf-8') as f:
        for doc in llama_parse_documents:
            f.write(doc.text.replace('\n', '  \n'))
    
    from langchain_core.documents import Document
    documents = []
    for doc in llama_parse_documents:
        documents.append(
            Document(
                page_content=doc.text,
                metadata={
                    "source": "new_Law.pdf",
                    "page": doc.metadata.get("page_label", 0),  # Giữ số trang
                    "section": doc.metadata.get("section", ""),
                }
            )
        )
    
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators = [
            "\n\n===== Page",  # Phân tách theo trang
            "\n\nChapter",    # Phân tách theo chương
            "\n\nArticle",     # Phân tách theo điều
            "\n\nSection",     # Phân tách theo mục
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )
    splits = text_splitter.split_documents(documents)
    
    
    # Initialize Embeddings
    embeddings = FastEmbedEmbeddings()
    
    # Create and persist a Chroma vector database from the chunked documents
    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        url=qdrant_url,
        collection_name="deeplaw",
    )
    
    print('Vector DB created successfully !')


if __name__ == "__main__":
    create_vector_database()