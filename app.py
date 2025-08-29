import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
import google.generativeai as genai
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()
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

uploaded = st.file_uploader(
    "Upload 3-5 text files (.txt)", type=["txt"], accept_multiple_files=True
)

if uploaded:
    # Read uploaded docs
    docs = [f.read().decode("utf-8") for f in uploaded]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for d in docs:
        chunks.extend(splitter.create_documents([d]))

    # Build embeddings + Chroma
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma.from_documents(
        chunks, embeddings, persist_directory="chroma_db"
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Build Retrieval QA
    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    # ——— Tools ———
    def rag_tool(query: str) -> str:
        """Try RAG; fallback to Gemini global knowledge if irrelevant."""
        res = qa.invoke(query)
        st.session_state["last_sources"] = res.get("source_documents", [])

        # If no relevant docs, fallback to Gemini directly
        if not st.session_state["last_sources"] or all(
            not doc.page_content.strip() for doc in st.session_state["last_sources"]
        ):
            llm_resp = llm.invoke(query)
            return llm_resp.content

        return res["result"]

    def calc(expr: str) -> str:
        try:
            return str(eval(expr))
        except:
            return "Invalid calculation"

    def define(word: str) -> str:
        return f"Definition (placeholder) for: {word}"

    tools = [
        Tool(
            name="RAG Q&A",
            func=rag_tool,
            description="Answer from your uploaded text if relevant, else global knowledge",
        ),
        Tool(
            name="Calculator",
            func=calc,
            description="Calculate if query includes 'calculate'",
        ),
        Tool(
            name="Dictionary",
            func=define,
            description="Define if query includes 'define'",
        ),
    ]

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    # ——— Query UI ———
    query = st.text_input("Ask here:")
    if query:
        with st.spinner("Thinking..."):
            resp = agent.invoke({"input": query})

        st.write("### 🤖 Answer:")
        st.write(resp["output"])

        # Show sources if available
        if "last_sources" in st.session_state and st.session_state["last_sources"]:
            st.write("### 📑 Sources:")
            for i, doc in enumerate(st.session_state["last_sources"], 1):
                content = doc.page_content if doc.page_content else "No content"
                st.markdown(f"**Source {i}:** {content[:300]}...")
