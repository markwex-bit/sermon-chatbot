import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 🔐 Set your OpenAI API key directly
with open("openai_key.txt") as f:
    openai_api_key = f.read().strip()

# Load the vector store
db = FAISS.load_local(
    "sermon_index",
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    allow_dangerous_deserialization=True
)

# Setup the retriever
retriever = db.as_retriever()

# Setup the LLM with retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
    retriever=retriever,
    return_source_documents=True
)

print("🧠 Chatbot ready. Ask a question based on your devotions.\n(Type 'exit' to quit)\n")

# Interactive loop
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain(query)
    print(f"\n💬 Chatbot: {result['result']}\n")

    print("📚 Sources:")
    for doc in result['source_documents']:
        snippet = doc.page_content[:200].strip().replace('\n', ' ')
        source = doc.metadata.get('source', 'Unknown file')
        print(f"🔹 {os.path.basename(source)}: {snippet}...\n")
