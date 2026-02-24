"""
Ejemplo Avanzado: RAG con Validación y Control de Calidad

Este script demuestra patrones más sofisticados para producción:
- Validación de relevancia
- Manejo explícito de errores
- Logging y monitoreo
- Configuración flexible
- Cache de resultados
"""

import os
import getpass
import json
import time
from typing import Optional
import bs4

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.tools import tool


class Config:
    """Configuración centralizada del RAG"""
    
    # Modelo y embeddings
    LLM_MODEL = "gpt-4.1"
    EMBEDDING_MODEL = "text-embedding-3-large"
    
    # Parámetros de texto
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MIN_RELEVANCE_SCORE = 0.7
    
    # Recuperación
    MAX_DOCUMENTS = 3
    
    # Fuentes de datos
    SOURCES = [
        "https://lilianweng.github.io/posts/2023-06-23-agent",
    ]
    
    @classmethod
    def from_env(cls):
        """Cargar configuración desde variables de entorno"""
        cls.LLM_MODEL = os.getenv("LLM_MODEL", cls.LLM_MODEL)
        cls.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", cls.CHUNK_SIZE))
        return cls


class RAGSystem:
    """Sistema RAG robusto con validación y logging"""
    
    def __init__(self, config: Config = None):
        """
        Inicializar el sistema RAG
        
        Args:
            config: Objeto de configuración
        """
        self.config = config or Config()
        self.model = None
        self.embeddings = None
        self.vector_store = None
        self.agent = None
        self.stats = {
            "docs_loaded": 0,
            "chunks_created": 0,
            "queries_processed": 0,
            "avg_query_time": 0,
        }
        
    def setup(self) -> bool:
        """
        Configurar el sistema RAG
        
        Returns:
            True si la configuración fue exitosa, False en caso contrario
        """
        try:
            print("Configurando sistema RAG...")
            
            # Verificar API key
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = getpass.getpass(
                    "Introduce tu API key de OpenAI: "
                )
            
            # Inicializar modelo
            print(f"  - Inicializando modelo {self.config.LLM_MODEL}...")
            self.model = init_chat_model(self.config.LLM_MODEL)
            
            # Inicializar embeddings
            print(f"  - Inicializando embeddings {self.config.EMBEDDING_MODEL}...")
            self.embeddings = OpenAIEmbeddings(model=self.config.EMBEDDING_MODEL)
            
            # Crear vector store
            print("  - Creando vector store...")
            self.vector_store = InMemoryVectorStore(self.embeddings)
            
            # Cargar documentos
            self._load_documents()
            
            # Crear agente
            self._create_agent()
            
            print(f"Sistema configurado exitosamente!")
            print(f"  - Documentos cargados: {self.stats['docs_loaded']}")
            print(f"  - Chunks creados: {self.stats['chunks_created']}")
            
            return True
            
        except Exception as e:
            print(f"Error durante configuración: {e}")
            return False
    
    def _load_documents(self):
        """Cargar documentos desde las fuentes configuradas"""
        print("Cargando documentos...")
        
        for source in self.config.SOURCES:
            try:
                print(f"  Descargando: {source}")
                
                bs4_strainer = bs4.SoupStrainer(
                    class_=("post-title", "post-header", "post-content")
                )
                loader = WebBaseLoader(
                    web_paths=(source,),
                    bs_kwargs={"parse_only": bs4_strainer},
                )
                docs = loader.load()
                
                self.stats['docs_loaded'] += len(docs)
                
                # División en chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.CHUNK_SIZE,
                    chunk_overlap=self.config.CHUNK_OVERLAP,
                    add_start_index=True,
                )
                splits = text_splitter.split_documents(docs)
                
                self.stats['chunks_created'] += len(splits)
                
                # Agregar al vector store
                self.vector_store.add_documents(splits)
                
                print(f"  OK: {len(docs)} documentos, {len(splits)} chunks")
                
            except Exception as e:
                print(f"  ERROR cargando {source}: {e}")
    
    def _create_agent(self):
        """Crear el agente con herramientas"""
        
        @tool(response_format="content_and_artifact")
        def retrieve_context_with_validation(query: str):
            """Recupera información con validación de relevancia"""
            try:
                # Búsqueda con scores
                results = self.vector_store.similarity_search_with_scores(
                    query,
                    k=self.config.MAX_DOCUMENTS
                )
                
                if not results:
                    return "No se encontraron documentos relevantes."
                
                # Filtrar por score mínimo
                relevant_results = [
                    (doc, score) for doc, score in results
                    if score >= self.config.MIN_RELEVANCE_SCORE
                ]
                
                if not relevant_results:
                    return (
                        f"No se encontraron documentos con relevancia > "
                        f"{self.config.MIN_RELEVANCE_SCORE}"
                    )
                
                # Formatear resultados
                formatted_results = []
                for doc, score in relevant_results:
                    formatted_results.append(
                        f"[Relevancia: {score:.3f}]\n"
                        f"Fuente: {doc.metadata}\n"
                        f"Contenido: {doc.page_content[:500]}...\n"
                    )
                
                serialized = "\n".join(formatted_results)
                return serialized, relevant_results
                
            except Exception as e:
                return f"Error durante recuperación: {e}"
        
        tools = [retrieve_context_with_validation]
        prompt = (
            "Eres un asistente experto en Inteligencia Artificial. "
            "Tienes acceso a una base de conocimiento sobre agentes de IA y "
            "técnicas de procesamiento de lenguaje natural. "
            "Utiliza la herramienta de recuperación para responder preguntas basándote "
            "en información verificable. Si no encuentras información relevante, "
            "comunícalo claramente al usuario."
        )
        
        self.agent = create_agent(
            self.model,
            tools,
            system_prompt=prompt
        )
    
    def query(self, question: str, stream: bool = True) -> dict:
        """
        Procesar una consulta
        
        Args:
            question: Pregunta del usuario
            stream: Si True, muestra respuesta en tiempo real
            
        Returns:
            Diccionario con resultados y metadatos
        """
        if not self.agent:
            return {"error": "Sistema no configurado"}
        
        start_time = time.time()
        self.stats['queries_processed'] += 1
        
        try:
            results = []
            
            if stream:
                print("\n" + "=" * 80)
                print(f"PREGUNTA: {question}")
                print("=" * 80 + "\n")
                
                for event in self.agent.stream(
                    {"messages": [{"role": "user", "content": question}]},
                    stream_mode="values"
                ):
                    if "messages" in event:
                        last_message = event["messages"][-1]
                        if hasattr(last_message, 'content'):
                            results.append(last_message.content)
                        last_message.pretty_print()
            else:
                response = self.agent.invoke(
                    {"messages": [{"role": "user", "content": question}]}
                )
                results = [response["messages"][-1].content]
            
            elapsed_time = time.time() - start_time
            
            # Actualizar promedio de tiempo
            if self.stats['queries_processed'] > 0:
                self.stats['avg_query_time'] = (
                    (self.stats['avg_query_time'] * 
                     (self.stats['queries_processed'] - 1) + elapsed_time) /
                    self.stats['queries_processed']
                )
            
            return {
                "status": "success",
                "question": question,
                "response": results,
                "time_elapsed": elapsed_time,
            }
            
        except Exception as e:
            return {
                "status": "error",
                "question": question,
                "error": str(e),
            }
    
    def get_stats(self) -> dict:
        """Obtener estadísticas del sistema"""
        return {
            **self.stats,
            "vector_store_size": len(
                self.vector_store._collection._documents
                if hasattr(self.vector_store, '_collection') else 0
            ),
        }


def main():
    """Función principal de demostración"""
    
    # Crear sistema RAG
    rag = RAGSystem(Config())
    
    # Configurar
    if not rag.setup():
        print("Fallo al configurar el sistema")
        return
    
    # Preguntas de ejemplo
    questions = [
        "¿Qué es la descomposición de tareas en agentes LLM?",
        "¿Cómo funcionan los agentes autónomos?",
        "¿Cuáles son las limitaciones de los agentes actuales?",
    ]
    
    # Procesar preguntas
    for question in questions:
        result = rag.query(question, stream=True)
        
        if result['status'] == 'error':
            print(f"Error: {result['error']}")
        else:
            print(f"Tiempo de respuesta: {result['time_elapsed']:.2f}s\n")
    
    # Mostrar estadísticas
    print("\n" + "=" * 80)
    print("ESTADISTICAS DEL SISTEMA")
    print("=" * 80)
    stats = rag.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
