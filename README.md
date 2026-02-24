# Sistema RAG - Retrieval-Augmented Generation

## Descripción General

Este proyecto implementa un Sistema de Generación Aumentada por Recuperación (RAG - Retrieval-Augmented Generation). RAG es un enfoque que combina búsqueda de información relevante con generación de texto basada en modelos de lenguaje grande (LLM), mejorando la precisión y relevancia de las respuestas.

## Teoría Base

### ¿Qué es RAG?

RAG resuelve un problema común en LLMs: la falta de información actualizada o específica sobre dominios particulares. En lugar de entrenar o hacer fine-tuning del modelo (operación cara), RAG:

1. **Recupera** documentos relevantes de una base de conocimiento
2. **Aumenta** el prompt del modelo con esos documentos
3. **Genera** respuestas basadas en el contexto recuperado

Este enfoque es especialmente útil para:
- Consultas sobre documentos específicos
- Información actualizada (el modelo puede acceder a contenido reciente)
- Reducción de alucinaciones (las respuestas se basan en fuentes verificables)

### Componentes Clave de RAG

1. **Embeddings**: Representación vectorial de texto en un espacio multidimensional
2. **Vector Store**: Base de datos que almacena y busca vectores por similitud
3. **Recuperación**: Búsqueda por similitud para encontrar documentos relevantes
4. **Generación**: El LLM produce respuestas basadas en documentos recuperados

## Estructura del Código

### 1. Configuración Inicial

```python
import os
import getpass
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
```

- **Autenticación**: Si no existe la variable de entorno `OPENAI_API_KEY`, el script solicita la clave interactivamente
- **Embeddings**: Se utiliza el modelo `text-embedding-3-large` de OpenAI para convertir texto en vectores
- **Vector Store**: Se usa un almacén en memoria para agilidad en desarrollo

### 2. Carga de Documentos

```python
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
```

**Proceso:**
- **WebBaseLoader**: Descarga contenido HTML de URLs específicas
- **BeautifulSoup Strainer**: Filtra solo las secciones relevantes (título, encabezados, contenido)
- **Resultado**: Lista de documentos con contenido y metadatos

### 3. División de Documentos en Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
splits = text_splitter.split_documents(docs)
```

**Propósito**: Los documentos completos son muy largos para procesar eficientemente.

**Parámetros:**
- `chunk_size=1000`: Cada fragmento contiene máximo 1000 caracteres
- `chunk_overlap=200`: Los fragmentos se solapan en 200 caracteres para mantener contexto
- `add_start_index=True`: Registra la posición original de cada fragmento

**Utilidad del solapamiento**: Evita que información importante se pierda en los límites de chunks

### 4. Indexación en Vector Store

```python
vector_store.add_documents(splits)
```

**Lo que sucede internamente:**
1. Cada fragmento se convierte en un vector usando embeddings
2. Los vectores se almacenan en el vector store
3. Se crea un índice para búsqueda rápida por similitud

### 5. Herramienta de Recuperación

```python
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Recupera información relevante para responder una consulta."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
```

**Funcionamiento:**
- Convierte la consulta en vector (usando el mismo modelo de embeddings)
- Busca los k=2 documentos más similares en el vector store
- Formatea los resultados con metadatos y contenido
- Devuelve contexto para que el agente lo use

**Métrica de similitud**: Generalmente similitud del coseno entre vectores

### 6. Creación del Agente

```python
agent = create_agent(model, tools, system_prompt=prompt)
```

**Arquitectura:**
- El modelo base es GPT-4.1
- El agente tiene acceso a la herramienta `retrieve_context`
- El prompt del sistema instruye al agente sobre cuándo usar la herramienta

**Flujo de ejecución:**
1. Recibe una pregunta del usuario
2. Decide si necesita recuperar contexto
3. Llama a `retrieve_context` si es necesario
4. Genera una respuesta basada en el contexto recuperado

### 7. Ejecución

```python
query = "¿Qué es la descomposición de tareas en agentes LLM?"
for event in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

**Parámetros:**
- `stream_mode="values"`: Devuelve el estado completo de mensajes en cada paso
- `pretty_print()`: Formatea la salida de manera legible

**Ventaja del streaming**: Se pueden ver los pasos intermediaos del agente en tiempo real

## Flujo Completo de Ejecución

1. Usuario pregunta: "¿Qué es la descomposición de tareas en agentes LLM?"
2. Agente convierte pregunta en vector
3. Agente busca documentos similares (recuperación)
4. Se agregan los documentos al contexto
5. LLM genera respuesta utilizando:
   - Instrucciones del sistema
   - Pregunta del usuario
   - Contexto recuperado

## Requisitos Técnicos

- Python 3.8+
- OpenAI API key
- Librerías: langchain, langchain-openai, beautifulsoup4

## Instalación

```bash
pip install langchain langchain-openai langchain-community beautifulsoup4
```

## Variables de Entorno

Configura tu clave de API de OpenAI:
```bash
export OPENAI_API_KEY="tu-clave-aqui"
```

O permite que el script la solicite interactivamente al ejecutarse.

## Casos de Uso

1. **Sistemas de soporte de atención al cliente**: Responder basándose en documentación específica
2. **Análisis de datos financieros**: Recuperar información de reportes actualizados
3. **Asistentes de investigación**: Buscar papers científicos relevantes
4. **Sistemas de preguntas y respuestas**: Q&A sobre bases de conocimiento especificadas
5. **Chatbots especializados**: Comportamiento específico del dominio sin reentrenamiento

## Ventajas del Enfoque RAG

- Flexibilidad: Fácil de actualizar la base de conocimiento
- Transparencia: Las respuestas citan las fuentes
- Eficiencia: No requiere reentrenamiento del modelo
- Precisión: Reduce alucinaciones usando información verificable
- Escalabilidad: Funciona con múltiples fuentes de datos
