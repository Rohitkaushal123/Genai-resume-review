import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import streamlit as st

# Load GROQ
groq_api = os.environ['GROQ_API_KEY']
llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api)

st.title("🤖 GenAI Resume Reviewer & Insight Generator")
st.write("📤 Upload your resume (PDF) to get AI-generated insights")

uploaded_file = st.file_uploader("Choose a PDF resume", type="pdf")

# Process resume only once on upload
if uploaded_file is not None and "retriever" not in st.session_state:
    with st.spinner("🚀 Firing up the AI engines to decode your resume..."):
        with open("temp_resume.pdf", 'wb') as f:
            f.write(uploaded_file.read())

        loader = PDFPlumberLoader("temp_resume.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embedding)

        # Cache retriever
        st.session_state.retriever = vectordb.as_retriever()

# Ask user input
question = st.text_input("💬 Ask something about your resume (e.g., 'What can I improve?')")

# Prompt template
prompt = PromptTemplate(
    template="""
    You are an AI career expert. Give feedback on the resume based on the following content:

    {context}

    Question: {question}
    """,
    input_variables=['context', 'question']
)

# Button click: only runs feedback spinner
if st.button("🔍 Generate Feedback"):
    if not question:
        st.warning("❗ Please enter a question before generating feedback.")
    elif "retriever" not in st.session_state:
        st.error("❗ Please upload a valid resume first.")
    else:
        with st.spinner("🤖 Generating your resume feedback..."):
            retriever = st.session_state.retriever
            relevant_docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            final_prompt = prompt.format(context=context, question=question)
            response = llm.invoke(final_prompt)

        st.success("✅ Feedback generated!")
        st.subheader("🔍 AI Feedback:")
        st.write(response.content)
