import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
import google.generativeai as genai
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Access API key
api_key = os.getenv("GOOGLE_API_KEY")
print("API Key Loaded:", api_key is not None)

# ——— Streamlit UI Setup ———
st.title("📚 RAG Assistant with Gemini & ChromaDB")

# Configure Gemini API
genai.configure(api_key=api_key)

# Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key, 
    temperature=0,
)

uploaded = st.file_uploader("Upload 3-5 text files (.txt)", type=["txt"], accept_multiple_files=True)
if uploaded:
    docs = [f.read().decode("utf-8") for f in uploaded]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for d in docs:
        chunks.extend(splitter.create_documents([d]))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Build QA chain (returns both result + sources)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # ✅ Wrapper to only return "result" so it works with Tool
    def rag_tool(query: str) -> str:
        res = qa.invoke(query)
        # Save sources for later display
        st.session_state["last_sources"] = res.get("source_documents", [])
        return res["result"]

    # Simple tools
    def calc(expr: str) -> str:
        try:
            return str(eval(expr))
        except:
            return "Invalid calculation"

    def define(word: str) -> str:
        return f"Definition (placeholder) for: {word}"

    tools = [
        Tool(name="RAG Q&A", func=rag_tool, description="Answer from your uploaded text"),
        Tool(name="Calculator", func=calc, description="Calculate if query includes 'calculate'"),
        Tool(name="Dictionary", func=define, description="Define if query includes 'define'")
    ]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    query = st.text_input("Ask here:")
    if query:
        with st.spinner("Thinking..."):
            resp = agent.invoke(query)
        st.write("### 🤖 Answer:")
        st.write(resp["output"])

        # Show sources if available
        if "last_sources" in st.session_state and st.session_state["last_sources"]:
            st.write("### 📑 Sources:")
            for i, doc in enumerate(st.session_state["last_sources"], 1):
                st.markdown(f"**Source {i}:** {doc.page_content[:300]}...")
