# Documentación Técnica Detallada

## Tabla de Contenidos

1. [Arquitectura Interna de LangChain](#arquitectura-interna-de-langchain)
2. [Embeddings y Vector Search](#embeddings-y-vector-search)
3. [Proceso de Indexación](#proceso-de-indexación)
4. [Algoritmos de Recuperación](#algoritmos-de-recuperación)
5. [Optimización de Performance](#optimización-de-performance)
6. [Debugging y Logging](#debugging-y-logging)

## Arquitectura Interna de LangChain

### Componentes Principales

LangChain está organizado en varios módulos:

```
langchain/
├── llms/              # Interfaces para modelos de lenguaje
├── chat_models/       # Modelos específicos de chat
├── embeddings/        # Generadores de vectores
├── vectorstores/      # Almacenes de vectores
├── document_loaders/  # Cargadores de documentos
├── text_splitters/    # Divisores de texto
├── prompts/           # Gestión de prompts
├── tools/             # Herramientas para agentes
├── agents/            # Framework para agentes
└── chains/            # Cadenas de operaciones
```

### Ciclo de Vida de una Solicitud

```
1. Input del Usuario
          ↓
2. Prompt Template Rendering
   - Interpolar variables
   - Crear mensaje formateado
          ↓
3. Envío a LLM
   - Serializar prompt
   - Hacer llamada HTTP a API
   - Manejo de errores y reintentos
          ↓
4. Procesamiento de Respuesta
   - Parsear output del modelo
   - Extraer contenido relevante
          ↓
5. Devolución al Usuario
```

## Embeddings y Vector Search

### ¿Qué son los Embeddings?

Los embeddings son representaciones numéricas de texto en espacios de alta dimensión (típicamente 1536 dimensiones para text-embedding-3-large).

**Propiedades:**
- Vectores similares representan texto semánticamente similar
- Distancia euclidiana o similitud del coseno miden similitud
- Invariantes a pequeños cambios léxicos (sinónimos)

### Proceso de Embedding

```
Texto de entrada: "Los gatos son animales"
                        ↓
Tokenización: ["Los", "gatos", "son", "animales"]
                        ↓
Encoding Neural: extrae características semánticas
                        ↓
Vector de salida: [-0.012, 0.045, 0.123, ..., 0.089]
                (1536 dimensiones)
```

### Similitud del Coseno

La similitud del coseno es el método estándar para comparar embeddings:

```
similarity = (A · B) / (||A|| * ||B||)

Rango: -1 (opuesto) a 1 (idéntico)
Uso en RAG: > 0.7 típicamente indica relevancia
```

## Proceso de Indexación

### Paso 1: Carga de Documentos

El `WebBaseLoader` descarga HTML mediante:

```python
loader = WebBaseLoader(web_paths=("https://...",))
docs = loader.load()

# Resultado: List[Document]
# Cada Document tiene:
# - page_content: str (texto)
# - metadata: dict (fuente, URL, etc)
```

### Paso 2: División en Chunks

`RecursiveCharacterTextSplitter` divide recursivamente por:

1. Párrafos
2. Oraciones
3. Espacios
4. Caracteres

Con parámetro `chunk_overlap=200`:

```
Chunk 1: [Pos 0-1000]
Chunk 2: [Pos 800-1800]     <- Superpone 200 chars con Chunk 1
Chunk 3: [Pos 1600-2600]    <- Superpone 200 chars con Chunk 2
```

**Ventaja del solapamiento:** Preserva contexto en los límites

### Paso 3: Generación de Embeddings

```python
vector_store.add_documents(splits)

# Internamente:
for doc in splits:
    vector = embeddings.embed_query(doc.page_content)
    # Guardar (vector, doc) en índice
```

### Paso 4: Indexación

El vector store crea una estructura de datos que permite:
- Búsqueda rápida por similitud
- Búsqueda exacta por metadatos
- Recuperación de documentos originales

## Algoritmos de Recuperación

### Recuperación Semántica Simple (usado en el proyecto)

```python
def similarity_search(query, k=2):
    query_vector = embed(query)
    scores = [cosine_similarity(query_vector, doc_vector) 
              for doc_vector in index]
    top_k = sorted(scores)[:k]
    return [docs[i] for i in top_k indices]
```

**Complejidad:** O(n*d) donde n=documentos, d=dimensiones

### Alternativas Avanzadas

1. **Hybrid Search**: Combina búsqueda semántica con búsqueda de palabras clave
2. **MMR (Maximum Marginal Relevance)**: Evita duplicados conceptuales
3. **Búsqueda Filtrada**: Aplica filtros antes de buscar

## Optimización de Performance

### 1. Reducción de Dimensionalidad

Los modelos de embedding modernos permiten truncado:

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# 1536 -> 256 dimensiones manteniendo 99.96% del desempeño
```

### 2. Batching

Procesar múltiples documentos en una llamada API:

```python
# En lugar de:
for doc in docs:
    vector_store.add_document(doc)  # N llamadas

# Hacer:
vector_store.add_documents(docs)    # 1 llamada
```

### 3. Caching

Almacenar embeddings calculados previamente:

```python
from langchain.cache import SQLiteCache
import langchain
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
```

### 4. Vector Store Persistente

Para producción, usar bases de datos vectoriales:

```python
# Alternativa a InMemoryVectorStore:
from langchain_pinecone import PineconeVectorStore
from langchain_milvus import MilvusVectorStore
from langchain_weaviate import WeaviateVectorStore
```

## Debugging y Logging

### Habilitar Logging Detallado

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Ver todas las llamadas HTTP
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG)
```

### Herramientas de Debugging

#### 1. Inspeccionar un Prompt Renderizado

```python
messages = prompt.format_messages(
    tema="Python",
    pregunta="¿Qué es async?"
)
print(messages)
```

#### 2. Ver Embeddings

```python
vector = embeddings.embed_query("texto de prueba")
print(f"Dimensiones: {len(vector)}")
print(f"Primeras 5 componentes: {vector[:5]}")
```

#### 3. Verificar Recuperación

```python
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    docs = vector_store.similarity_search_with_scores(query, k=2)
    
    for doc, score in docs:
        print(f"Score: {score:.4f}")
        print(f"Content preview: {doc.page_content[:100]}...")
    
    # ... resto del código
```

#### 4. Rastrear Decisiones del Agente

```python
for event in agent.stream(config):
    if "tool" in event:
        print(f"Herramienta utilizada: {event['tool']}")
    if "result" in event:
        print(f"Resultado: {event['result']}")
```

## Casos Avanzados

### Extensión 1: Múltiples Vector Stores

```python
from langchain.vectorstores import [VectorStoreA, VectorStoreB]

store_a = VectorStoreA(embeddings)
store_b = VectorStoreB(embeddings)

store_a.add_documents(docs_financieros)
store_b.add_documents(docs_tecnicos)

@tool
def retrieve_financial(query):
    return store_a.similarity_search(query)

@tool
def retrieve_technical(query):
    return store_b.similarity_search(query)

tools = [retrieve_financial, retrieve_technical]
```

### Extensión 2: Validación de Relevancia

```python
@tool
def retrieve_context_filtered(query: str):
    docs = vector_store.similarity_search_with_scores(query, k=5)
    # Filtrar por score mínimo
    relevant_docs = [(doc, score) for doc, score in docs if score > 0.7]
    
    if not relevant_docs:
        return "No se encontró información relevante"
    
    serialized = "\n\n".join(...)
    return serialized
```

### Extensión 3: Re-ranking

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(model, llm_chain_prompt)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=vector_store.as_retriever(),
    base_compressor=compressor
)
```

## Métricas de Evaluación

### Para Recuperación

- **Precision@K**: De los K primeros docs, cuántos son relevantes
- **Recall@K**: De todos los docs relevantes, cuántos están en top K
- **MRR (Mean Reciprocal Rank)**: Posición promedio del primer doc relevante
- **NDCG (Normalized DCG)**: Ranking quality considerando orden

### Para Generación

- **BLEU Score**: Solapamiento n-grama con salida de referencia
- **ROUGE Score**: Cobertura de palabras/frases clave
- **BERTScore**: Similitud semántica usando embeddings
- **Human Evaluation**: Evaluación manual de calidad

## Troubleshooting Avanzado

### Problema: Embeddings lentos

**Causas:**
- Red lenta
- Demasiados documentos simultáneamente
- Timeout de API

**Soluciones:**
```python
# Aumentar timeout
embeddings = OpenAIEmbeddings(
    request_timeout=120
)

# Procesar en lotes pequeños
batch_size = 100
for i in range(0, len(docs), batch_size):
    vector_store.add_documents(docs[i:i+batch_size])
```

### Problema: Documentos recuperados irrelevantes

**Causas:**
- Chunks demasiado grandes/pequeños
- Query ambigua
- Vocabulario diferente al documentos

**Soluciones:**
```python
# Ajustar chunk_size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)

# Reformular query
reformulated_queries = [original_query, paraphrase1, paraphrase2]

# Búsqueda multi-query
all_docs = set()
for q in reformulated_queries:
    all_docs.update(vector_store.similarity_search(q))
```

### Problema: Agente se queda en loop infinito

**Soluciones:**
```python
# Limitar iteraciones
agent_with_timeout = agent.with_config(
    {"recursion_limit": 10}
)

# Instruir en el prompt
system_prompt = """
Eres un agente útil. 
Máximo 3 usos de herramientas.
Si no encuentras información, di "No se encontró información".
"""
```

## Referencias Técnicas

- [OpenAI Embedding Models](https://platform.openai.com/docs/guides/embeddings)
- [LangChain API Reference](https://python.langchain.com/api_reference/)
- [Vector Database Comparison](https://github.com/erikbern/ann-benchmarks)
- [Semantic Search Best Practices](https://huggingface.co/blog/semantic-search-hf)
