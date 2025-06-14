import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔐 Set your OpenAI API key here
with open("openai_key.txt") as f:
    openai_api_key = f.read().strip()

# Define folders
sermon_folder = os.path.join(os.path.dirname(__file__), "sermons")
vector_output_folder = os.path.join(os.path.dirname(__file__), "sermon_index")

# Load all sermon .txt files
all_docs = []
for filename in os.listdir(sermon_folder):
    if filename.endswith(".txt"):
        path = os.path.join(sermon_folder, filename)
        loader = TextLoader(path, encoding='utf-8')
        docs = loader.load()
        all_docs.extend(docs)

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(all_docs)

# Create vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(split_docs, embeddings)
db.save_local(vector_output_folder)

print("✅ Vector store rebuilt and saved to 'sermon_index/'")

