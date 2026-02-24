# Ejemplos Avanzados y Patrones de Implementación

Este documento contiene ejemplos prácticos adicionales y patrones para casos de uso específicos.

## Tabla de Contenidos

1. [RAG con Fuentes Múltiples](#rag-con-fuentes-múltiples)
2. [RAG con Procesamiento de PDFs](#rag-con-procesamiento-de-pdfs)
3. [Cadenas con Conversación](#cadenas-con-conversación)
4. [Búsqueda Inteligente](#búsqueda-inteligente)
5. [Validación y Guardrails](#validación-y-guardrails)

## RAG con Fuentes Múltiples

Procesar múltiples URLs y mantener separados los índices:

```python
from langchain_core.vectorstores import InMemoryVectorStore

sources = {
    "blog": "https://ejemplo.com/blog",
    "docs": "https://ejemplo.com/documentacion",
    "papers": "https://ejemplo.com/papers",
}

stores = {}
for name, url in sources.items():
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()
    
    # Agregar metadato de fuente
    for doc in docs:
        doc.metadata['source_type'] = name
    
    # Crear store
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    splits = splitter.split_documents(docs)
    stores[name] = InMemoryVectorStore(embeddings)
    stores[name].add_documents(splits)

# Usar en recuperación
@tool
def search_by_source(query: str, source: str):
    """Buscar en una fuente específica"""
    if source not in stores:
        return f"Fuente {source} no disponible"
    return stores[source].similarity_search(query, k=3)
```

## RAG con Procesamiento de PDFs

Procesar archivos PDF locales (requiere PyPDF2):

```bash
pip install PyPDF2
```

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cargar PDF
loader = PyPDFLoader("documento.pdf")
pages = loader.load()

# Dividir en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(pages)

# Indicar origen
for chunk in chunks:
    chunk.metadata['source'] = "documento.pdf"

# Indexar
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)

print(f"Indexados {len(chunks)} chunks del PDF")
```

## Cadenas con Conversación

Mantener contexto entre preguntas del usuario:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ChatMessage

class ConversationalRAG:
    def __init__(self, model, vector_store):
        self.model = model
        self.vector_store = vector_store
        self.conversation_history = []
    
    def answer(self, query: str) -> str:
        # Recuperar contexto
        docs = self.vector_store.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Crear prompt con historial
        messages = self.conversation_history + [
            ChatMessage(role="user", content=query)
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Contexto: {context}"),
            ("user", "{query}")
        ])
        
        # Generar respuesta
        response = self.model.invoke(
            prompt.format(query=query)
        )
        
        # Guardar en historial
        self.conversation_history.append(
            ChatMessage(role="user", content=query)
        )
        self.conversation_history.append(
            ChatMessage(role="assistant", content=response.content)
        )
        
        return response.content

# Uso
rag = ConversationalRAG(model, vector_store)
print(rag.answer("¿Qué es un agente?"))
print(rag.answer("¿Cómo funcionan?"))  # Tendrá contexto de la pregunta anterior
```

## Búsqueda Inteligente

Implementar búsqueda con reformulación y multi-query:

```python
from langchain.prompts import PromptTemplate

class IntelligentSearcher:
    def __init__(self, model, vector_store):
        self.model = model
        self.vector_store = vector_store
    
    def reformulate_query(self, query: str) -> list[str]:
        """Generar variaciones de la query para búsqueda más robusta"""
        
        prompt = PromptTemplate.from_template(
            """Genera 3 variaciones alternativas de la siguiente pregunta 
            para mejorar la búsqueda. Usa diferentes palabras clave y framing.
            
            Pregunta original: {query}
            
            Variaciones (una por línea):"""
        )
        
        result = self.model.invoke(prompt.format(query=query))
        variants = result.content.strip().split('\n')
        return [query] + variants
    
    def search(self, query: str, k: int = 3) -> list:
        """Búsqueda multi-query con deduplicación"""
        
        # Reformular
        queries = self.reformulate_query(query)
        
        # Buscar con todas las variaciones
        all_docs = {}
        for q in queries:
            docs = self.vector_store.similarity_search(q, k=k)
            for i, doc in enumerate(docs):
                key = doc.page_content[:100]  # Usar contenido como ID
                if key not in all_docs:
                    all_docs[key] = doc
        
        # Retornar documentos únicos
        return list(all_docs.values())[:k]

# Uso
searcher = IntelligentSearcher(model, vector_store)
results = searcher.search("¿Qué es descomposición de tareas?")
```

## Validación y Guardrails

Validar calidad de respuestas:

```python
from langchain.chains import LLMChain

class ValidatedRAG:
    def __init__(self, model, vector_store):
        self.model = model
        self.vector_store = vector_store
    
    def check_relevance(self, response: str, docs: list) -> dict:
        """Verificar que la respuesta está basada en los documentos"""
        
        verification_prompt = ChatPromptTemplate.from_template(
            """Dados estos documentos y una respuesta, 
            verifica si la respuesta está completamente basada en los documentos.
            
            Documentos:
            {docs}
            
            Respuesta:
            {response}
            
            Responde VÁLIDO o INVÁLIDO, seguido de una breve explicación."""
        )
        
        docs_text = "\n\n".join([d.page_content for d in docs])
        
        result = self.model.invoke(
            verification_prompt.format(
                docs=docs_text, 
                response=response
            )
        )
        
        is_valid = "VÁLIDO" in result.content.upper()
        return {
            "valid": is_valid,
            "explanation": result.content
        }
    
    def check_confidance_score(self, response: str) -> float:
        """Estimar confianza de la respuesta (0-1)"""
        
        score_prompt = ChatPromptTemplate.from_template(
            """Dado el siguiente texto de respuesta, 
            evalúa cuán confiada y bien fundamentada parece (0-1).
            
            Respuesta:
            {response}
            
            Responde solo un número entre 0 y 1."""
        )
        
        result = self.model.invoke(score_prompt.format(response=response))
        try:
            score = float(result.content.strip())
            return max(0, min(1, score))  # Clamp a 0-1
        except:
            return 0.5

# Uso
validator = ValidatedRAG(model, vector_store)

# Obtener respuesta
docs = vector_store.similarity_search("¿Qué es RAG?")
response = model.invoke("Basándote en esto... ¿Qué es RAG?")

# Validar
relevance = validator.check_relevance(response.content, docs)
confidence = validator.check_confidance_score(response.content)

print(f"Respuesta válida: {relevance['valid']}")
print(f"Explicación: {relevance['explanation']}")
print(f"Confianza: {confidence:.2%}")
```

## Caching de Embeddings

Mejorar performance cachando embeddings calculados:

```python
import json
from pathlib import Path

class CachedEmbeddingStore:
    def __init__(self, cache_file: str = "embeddings_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Cargar cache de disco"""
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Guardar cache a disco"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def get_embedding(self, text: str, embeddings) -> list:
        """Obtener embedding con cache"""
        
        key = hash(text)
        if key in self.cache:
            return self.cache[key]
        
        # Calcular
        embedding = embeddings.embed_query(text)
        self.cache[key] = embedding
        self._save_cache()
        
        return embedding

# Uso
cache = CachedEmbeddingStore()
embedding = cache.get_embedding("texto", embeddings)
```

## Logging y Monitoreo

Implementar logging completo:

```python
import logging
import json
from datetime import datetime

class RAGLogger:
    def __init__(self, log_file: str = "rag.log"):
        self.log_file = log_file
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Configurar logging"""
        logger = logging.getLogger("RAG")
        logger.setLevel(logging.DEBUG)
        
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_query(self, query: str, response: str, time_ms: float):
        """Log una consulta"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_length": len(response),
            "time_ms": time_ms,
            "type": "query"
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, query: str, error: str):
        """Log un error"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "error": error,
            "type": "error"
        }
        self.logger.error(json.dumps(log_entry))

# Uso
logger = RAGLogger()
logger.log_query("¿Qué es RAG?", "RAG es...", 1234.5)
```

## Integración con Base de Datos

Usar una base de datos vectorial persistente (ejemplo con Faiss):

```bash
pip install faiss-cpu  # o faiss-gpu
```

```python
from langchain_community.vectorstores import FAISS

# Crear índice
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Guardar a disco
vector_store.save_local("rag_index")

# Cargar desde disco
vector_store = FAISS.load_local("rag_index", embeddings)
```

---

## Más Patrones

Para más ejemplos, consulta:
- [Documentación Oficial de LangChain](https://python.langchain.com/)
- [Cookbook de LangChain](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangChain Templates](https://github.com/langchain-ai/langchain-templates)

¡Contribuye tus propios patrones! Ver [CONTRIBUTING.md](CONTRIBUTING.md)
