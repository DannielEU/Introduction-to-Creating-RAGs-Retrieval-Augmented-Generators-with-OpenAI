
import os
import getpass
import bs4

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.tools import tool


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Introduce tu API key de OpenAI: ")


model = init_chat_model("gpt-4.1")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)


bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
splits = text_splitter.split_documents(docs)


vector_store.add_documents(splits)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Recupera información relevante para responder una consulta."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]
prompt = (
    "Tienes acceso a una herramienta que recupera contexto de un blog. "
    "Utilízala para ayudar a responder preguntas de los usuarios."
)
agent = create_agent(model, tools, system_prompt=prompt)


query = "¿Qué es la descomposición de tareas en agentes LLM?"
for event in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
    event["messages"][-1].pretty_print()
