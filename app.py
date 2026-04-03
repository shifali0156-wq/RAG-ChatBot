from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import os 
import requests
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

def load_file(file_path):
  extension=file_path.split('.')[-1]
  if extension=="pdf":
    from langchain_community.document_loaders import PyPDFLoader
    loader=PyPDFLoader(file_path)
  elif extension=="docx":
    from langchain_community.document_loaders import Docx2txtLoader
    loader=Docx2txtLoader(file_path)
  elif extension=="txt":
    from langchain_community.document_loaders import TextLoader
    loader=TextLoader(file_path)
  elif extension=="pptx":
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    loader=UnstructuredPowerPointLoader(file_path)
  doc=loader.load()
  return doc


def build_chain_from_doc(file_path):
  doc=load_file(file_path)

  splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
  docs=splitter.split_documents(doc)

  embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  
  vector_store=Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="db"
  )
  vector_store.add_documents(docs)
  retriever=vector_store.as_retriever(search_type="similarity")
  return retriever


prompt=PromptTemplate(
    template="""You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.
      {context}
      Question: {question}""",
    input_variables=["context","question"]
)


def format_doc(docs):
  context="\n\n".join(doc.page_content for doc in docs)
  return context


def call_minimax(final_input):
    MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
    url = "https://api.minimax.io/v1/text/chatcompletion_v2"
    payload = {
        "model": "MiniMax-M2.7",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": final_input
            }
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_API_KEY}"
    }
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)


parser=StrOutputParser()

def create_chain(retriever):
  parallel_chain=RunnableParallel(
        {"context":retriever | RunnableLambda(format_doc),
        "question":RunnablePassthrough()}
  )
  chain1=prompt | RunnableLambda(lambda x: call_minimax(x.to_string())) | parser
  return parallel_chain | chain1


current_chain=None

def process_document(doc):
    global current_chain
    try:
        retriever = build_chain_from_doc(doc)
        current_chain = create_chain(retriever)
        return "Document processed! Now start chatting!"
    except Exception as e:
        return f"Error: {str(e)}"


def chat_fn(message, history):
    global current_chain
    if history is None:
        history=[]
    if current_chain is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Please process a document first."})
        return history
    try:
        answer = current_chain.invoke(message)
    except Exception as e:
        answer = f"Error: {str(e)}"
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history


with gr.Blocks() as demo:
    gr.Markdown("# Chat with Your Documents")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat")
            msg = gr.Textbox(placeholder="Ask something about the document...")
            msg.submit(chat_fn, inputs=[msg, chatbot], outputs=chatbot)
        with gr.Column(scale=1):
            gr.Markdown("### Upload Document")
            file_input = gr.File(label="Upload file")
            process_btn = gr.Button("Process Document")
            status = gr.Textbox(label="Status")
            process_btn.click(process_document, inputs=file_input, outputs=status)

demo.launch(
    theme=gr.themes.Citrus(),
    css="""
    body {
        background-color: #fff9db;
    }
    .gradio-container {
        background-color: #fff9db !important;
    }
    """
)
