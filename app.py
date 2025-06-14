import os
import csv
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import pygame
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# === Load OpenAI API key from file ===
with open("openai_key.txt", "r") as f:
    openai_api_key = f.read().strip()

# === Load vector store ===
db = FAISS.load_local(
    "sermon_index",
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever()

# === Setup QA chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
    retriever=retriever,
    return_source_documents=True
)

# === Streamlit App ===
st.set_page_config(page_title="Sermon Chatbot", layout="wide")
st.title("📖 Sermon Chatbot")
st.markdown("Ask a question based on your devotional files.")

# Collect user info
name = st.text_input("Name (optional)")
email = st.text_input("Email (optional)")
phone = st.text_input("Phone (optional)")

# Input box
query = st.text_input("Type your question or use voice input below:")

# Voice input
if st.button("🎤 Speak"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... speak now")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        st.success(f"You said: {query}")
    except sr.UnknownValueError:
        st.error("Sorry, could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Voice recognition failed: {e}")

# Source display toggle
show_sources = st.checkbox("Show source info", value=True)

# Run query
if query:
    with st.spinner("Thinking..."):
        result = qa_chain(query)
        st.markdown(f"**💬 Chatbot:** {result['result']}")

        # Text-to-speech output using gTTS and pygame
        try:
            tts = gTTS(text=result['result'], lang='en')
            audio_file = "response.mp3"
            tts.save(audio_file)
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
        except Exception as e:
            st.warning(f"Text-to-speech failed: {e}")

        # Show sources
        if show_sources:
            st.markdown("---")
            st.subheader("📂 Sources")
            for doc in result["source_documents"]:
                st.markdown(f"**File:** `{os.path.basename(doc.metadata['source'])}`")
                st.text(doc.page_content[:400].strip() + "...")

        # === Logging ===
        log_file = "chat_log.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sources = "; ".join([os.path.basename(doc.metadata["source"]) for doc in result["source_documents"]])

        # Create log file with header if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Name", "Email", "Phone", "Question", "Answer", "Sources"])

        # Append log entry
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name, email, phone, query, result["result"], sources])
