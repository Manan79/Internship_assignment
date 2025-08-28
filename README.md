
# 📘 Retrieval-Augmented Generation (RAG) Project

## 🔹 Overview

This project implements a **Retrieval-Augmented Generation (RAG)**
system using **LangChain, ChromaDB, Hugging Face embeddings, and Google
Generative AI (Gemini)**.

RAG enhances LLMs by combining them with external knowledge sources,
enabling: - Better **factual accuracy** - Up-to-date **context-aware
answers** - Handling of **large custom datasets**

------------------------------------------------------------------------

## 🚀 Features

-   Document ingestion and text chunking\
-   Embedding generation with **HuggingFace Transformers**\
-   Vector storage & retrieval using **Chroma**\
-   Question Answering powered by **Google Generative AI (Gemini)**\
-   Simple **Streamlit UI** for interaction

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── app.py              # Main Streamlit app
    ├── requirements.txt    # Dependencies
    ├── .env                # API keys (not committed to GitHub)
    └── README.md           # Project documentation

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone Repository

``` bash
git clone https://github.com/your-username/rag-project.git
cd rag-project
```

### 2. Create Virtual Environment

``` bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.\.venv\Scriptsctivate    # Windows
```

### 3. Install Requirements

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🔑 Setup API Keys

Create a `.env` file in the root folder:

    GOOGLE_API_KEY=your_google_api_key_here

------------------------------------------------------------------------

## ▶️ Run the App

``` bash
streamlit run app.py
```

Then open the provided **local URL** in your browser.

🌍 Or directly use the hosted app here:\
👉 [Streamlit App
Deployment](https://internshipassignment-zbzejjudwvsphc5zrcog8n.streamlit.app/)

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   **Python**
-   **LangChain**
-   **ChromaDB** (vector storage)
-   **HuggingFace Transformers**
-   **Google Generative AI (Gemini)**
-   **Streamlit**

------------------------------------------------------------------------

## 📸 Demo Screenshot

<img width="1430" height="674" alt="image" src="https://github.com/user-attachments/assets/6634107b-4182-4d20-b938-d70001cde3f5" />


------------------------------------------------------------------------

## 🔮 Future Improvements

-   Add support for multiple embedding models\
-   Enhance UI with conversation history\
-   Deploy on **Streamlit Cloud**

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to
discuss what you'd like to change.

------------------------------------------------------------------------

## 👤 Author

**Manan Sood**\
📌 [LinkedIn Profile](https://www.linkedin.com/in/mannan-sood-a38688253/)
=======


Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to change.

📜 License

This project is licensed under the MIT License.
>>>>>>> fd53a223f9bd4057550690600b8a88107eb91659
