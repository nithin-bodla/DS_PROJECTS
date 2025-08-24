import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

# --- Setup ---
api_key = os.environ["GOOGLE_API_KEY"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

def get_vector_store(text_chunks):
    ensure_event_loop()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question based on the context below. 
    If the answer is not in the context, say "I don't know."

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def user_input(user_question, db):
    docs = db.similarity_search(user_question)
    context = " ".join([d.page_content for d in docs])  
    chain = get_conversational_chain()
    response = chain({"context": context, "question": user_question})
    st.write("**Answer:**", response["text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini 💁")

    # Use Streamlit session state to persist FAISS
    if "db" not in st.session_state:
        st.session_state.db = None

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and click Submit & Process",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.db = get_vector_store(text_chunks)
                    st.success("✅ PDF Processed. You can now ask questions.")
            else:
                st.error("⚠️ Please upload at least one PDF.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        if st.session_state.db is None:
            st.error("⚠️ Please upload and process a PDF first.")
        else:
            user_input(user_question, st.session_state.db)

if __name__ == "__main__":
    main()
