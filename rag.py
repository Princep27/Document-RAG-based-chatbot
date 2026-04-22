from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("YOUR_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# Load documents
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS persistence
INDEX_PATH = "faiss_index"

if os.path.exists(INDEX_PATH):
    db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_PATH)

retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM (Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use context when needed."),
    ("human",
     "Context:\n{context}\n\nHistory:\n{history}\n\nQuestion: {question}")
])

# lightweight in-memory chat history (works per session)
chat_history = []
MAX_HISTORY = 6


def rag_chat(query: str):
    global chat_history

    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    sources = "\n---\n".join(d.page_content[:120] for d in docs)

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "history": history_text,
        "question": query
    })

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))
    chat_history = chat_history[-MAX_HISTORY:]

    return response.content, sources
