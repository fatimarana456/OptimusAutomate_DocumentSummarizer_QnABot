import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Document Summarizer & Q&A Bot",
    page_icon="📄",
    layout="wide"
)

# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

client = Groq(api_key=api_key)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📄 Document Summarizer & Q&A Bot")

st.markdown(
    "Upload a PDF document, generate a summary, and ask questions about its content."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📋 Features")

    st.markdown("""
    - PDF Upload
    - AI Summary
    - Question Answering
    - RAG Search
    - FAISS Vector Store
    - Groq LLM
    """)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "summary" not in st.session_state:
    st.session_state.summary = ""

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )

        st.session_state.vector_db = vector_db

    st.success("✅ PDF processed successfully")

# --------------------------------------------------
# SUMMARY SECTION
# --------------------------------------------------

if st.session_state.vector_db:

    st.divider()

    st.subheader("📝 Generate Summary")

    if st.button(
        "Generate Summary",
        use_container_width=True
    ):

        with st.spinner("Generating summary..."):

            docs = st.session_state.vector_db.similarity_search(
                "Provide a complete summary",
                k=10
            )

            context = "\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
            Summarize the following document clearly.

            Document:
            {context}

            Include:
            - Main topic
            - Key points
            - Important conclusions
            """

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=800
            )

            st.session_state.summary = (
                response.choices[0].message.content
            )

# --------------------------------------------------
# DISPLAY SUMMARY
# --------------------------------------------------

if st.session_state.summary:

    st.subheader("📌 Summary")

    st.info(st.session_state.summary)

# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

if st.session_state.vector_db:

    st.divider()

    st.subheader("🤖 Ask Questions")

    question = st.text_input(
        "Ask anything about the uploaded document"
    )

    if st.button(
        "Get Answer",
        use_container_width=True
    ):

        if question:

            with st.spinner("Searching document..."):

                docs = st.session_state.vector_db.similarity_search(
                    question,
                    k=5
                )

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                prompt = f"""
                You are a document assistant.

                Answer ONLY using the provided context.

                Context:
                {context}

                Question:
                {question}

                If the answer is not found in the context,
                say:
                "The document does not contain this information."
                """

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2,
                    max_tokens=600
                )

                answer = response.choices[0].message.content

                st.subheader("💡 Answer")

                st.write(answer)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built using Streamlit + Groq + LangChain + FAISS"
)