
# 📄 RAG Chatbot – Chat with Your Documents
### An AI-powered chatbot that allows users to upload documents and interact with them conversationally using a Retrieval-Augmented Generation (RAG) pipeline.

🔗 Live Demo: https://paru-73-rag-chatbot.hf.space


## Features

- Supports multiple formats: PDF, DOCX, TXT, PPTX
- Semantic search using vector embeddings
- Context-aware answers based only on document content
- Interactive chat interface
- Real-time document processing


## Tech Stack

**Language:** Python

**Framework:** Langchain

**Embeddings:** Hugging Face

**LLM:** MiniMax API

**Frontend/UI:** Gradio

**Deployment:** Hugging Face Spaces

## How it Works

**1.** Upload a document

**2.** Text is split into smaller chunks

**3.** Chunks are converted into embeddings

**4.** Stored in a vector database

**5.** User asks a question

**6.** Relevant chunks are retrieved

**7.** LLM generates a response based on context

## Installation


```bash
  git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
  cd rag-chatbot
  pip install -r requirements.txt
  python app.py
```
    
## Environment Variables

Create a .env file or set environment variables:

`MINIMAX_API_KEY=your_api_key`




## Project Structure

```bash
.
├── app.py
├── requirements.txt
└── README.md
```
## Deployment

Deployed on Hugging Face Spaces for easy public access.


## Contributing

Feel free to fork the repo and improve it. Contributions are always welcome!
