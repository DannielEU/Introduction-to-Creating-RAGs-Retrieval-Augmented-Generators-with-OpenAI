# LangChain: Fundamentos y RAG Avanzado

Aprende a construir aplicaciones sofisticadas con Large Language Models, desde cadenas simples hasta sistemas inteligentes de recuperación y generación.

**Contenido:**
- LangChain LLM Chain: Conceptos fundamentales
- Sistema RAG: Retrieval-Augmented Generation
- Ejemplos prácticos en tres niveles: básico, intermedio y avanzado

**Estado:** Proyecto educativo completo con documentación detallada
**Requisitos:** Python 3.8+, OpenAI API key
**Licencia:** Educativo

## Descripción General

Este proyecto contiene dos componentes principales que cubren el aprendizaje completo de LangChain:

1. **LangChain LLM Chain Básico** (llm_chain_example.py): Introducción a los conceptos fundamentales de LangChain
2. **Sistema RAG Avanzado** (rag.py): Implementación completa de Retrieval-Augmented Generation

Ambos componentes demuestran cómo construir aplicaciones sofisticadas de LLMs que van desde cadenas simples hasta sistemas inteligentes capaces de recuperar información contextual.

## Documentación del Proyecto

El proyecto incluye varios documentos para diferentes necesidades:

| Documento | Para | Duración |
|-----------|------|----------|
| [QUICKSTART.md](QUICKSTART.md) | Empezar inmediatamente | 5 min |
| [README.md](README.md) (este archivo) | Comprensión completa | 30 min |
| [EXAMPLES.md](EXAMPLES.md) | Patrones avanzados | 20 min |
| [DOCUMENTATION.md](DOCUMENTATION.md) | Detalles técnicos profundos | 1 hora |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribuir al proyecto | 10 min |

**Recomendación de Orden de Lectura:**
1. Comenzar: [QUICKSTART.md](QUICKSTART.md)
2. Aprender: [README.md](README.md) (este archivo)
3. Practicar: [EXAMPLES.md](EXAMPLES.md)
4. Profundizar: [DOCUMENTATION.md](DOCUMENTATION.md)
5. Contribuir: [CONTRIBUTING.md](CONTRIBUTING.md)

## Tabla de Contenidos

1. [Documentación del Proyecto](#documentación-del-proyecto)
2. [Inicio Rápido](#inicio-rápido)
3. [Conceptos Fundamentales](#parte-1-langchain-llm-chain-fundamentals)
4. [Teoría de RAG](#parte-2-sistema-rag---retrieval-augmented-generation)
5. [Instalación Detallada](#instalación-paso-a-paso)
6. [Ejemplos de Uso](#ejecución-de-los-ejemplos)
7. [Referencia Técnica](#requisitos-técnicos)
8. [Recursos Adicionales](#referencias)

## Inicio Rápido

Para empezar inmediatamente sin leer todo, sigue [QUICKSTART.md](QUICKSTART.md) (5 minutos).

Para desarrolladores experimentados:

```bash
# 1. Configurar entorno
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configurar API key (Windows)
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'tu-clave', 'User')

# 3. Ejecutar ejemplo
python rag.py
```

---

## Parte 1: LangChain LLM Chain Fundamentals

### ¿Qué es LangChain?

LangChain es un framework para desarrollar aplicaciones basadas en modelos de lenguaje grande (LLM). Proporciona abstracciones y herramientas para:

- Conectar LLMs con fuentes de datos externas
- Crear cadenas de operaciones complejas
- Construir agentes autónomos
- Gestionar memoria y contexto
- Implementar guardrails y validación

### Conceptos Básicos

#### 1. ChatPromptTemplate

Un template es una forma estructurada de crear prompts con variables dinámicas.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en {tema}"),
    ("user", "{pregunta}")
])
```

**Ventajas:**
- Reutilización de prompts
- Variables parametrizadas
- Separación clara de roles (system, user, assistant)

#### 2. LLM Chains (Cadenas)

Una cadena conecta un prompt con un modelo usando el operador pipe (`|`):

```python
chain = prompt | model
resultado = chain.invoke({"tema": "Python", "pregunta": "¿Qué es async?"})
```

**Flujo:**
1. Input se rellena en el template
2. Prompt relleno se envía al modelo
3. Modelo genera la respuesta

#### 3. Encadenamiento Múltiple

Puedes conectar múltiples cadenas secuencialmente:

```python
chain1 = prompt_titulo | model
chain2 = prompt_contenido | model

# Usar salida de chain1 como entrada de chain2
resultado_final = chain2.invoke({"titulo": chain1.invoke(...).content})
```

### Ejemplo Práctico: Archivo llm_chain_example.py

Este archivo demuestra:

- **Inicialización del modelo** (GPT-4.1)
- **Creación de templates de prompt** con variables dinámicas
- **Invocación de cadenas** con diferentes inputs
- **Encadenamiento múltiple** de operaciones sucesivas

**Ejecución:**
```bash
python llm_chain_example.py
```

**Salida esperada:**
Se genera una serie de respuestas del modelo para diferentes preguntas, demostrando cómo el mismo template puede reutilizarse con diferentes contextos.

---

## Parte 2: Sistema RAG - Retrieval-Augmented Generation

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

- Python 3.8 o superior
- OpenAI API key (obtener en https://platform.openai.com/api-keys)
- Pip (gestor de paquetes de Python)
- Conexión a internet (para descargar documentos y acceder a OpenAI API)

## Instalación Paso a Paso

### 1. Clonar o descargar el repositorio

```bash
# Si tienes git
git clone <url-del-repositorio>
cd <nombre-carpeta>

# O descarga el ZIP manualmente y extrae
```

### 2. Crear un entorno virtual

```bash
# En Windows
python -m venv .venv
.venv\Scripts\activate

# En macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**¿Por qué usar entorno virtual?**
- Aisla dependencias del proyecto
- Evita conflictos con otros proyectos
- Facilita compartir requisitos exactos

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install langchain langchain-openai langchain-community beautifulsoup4 requests
```

**Descripción de paquetes:**
- `langchain`: Framework principal
- `langchain-openai`: Integración con OpenAI
- `langchain-community`: Herramientas comunitarias
- `beautifulsoup4`: Parseo de HTML
- `requests`: Descargar contenido web

### 4. Configurar API Key

**Opción A: Variable de entorno persistente (Recomendado)**

En Windows PowerShell:
```powershell
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'tu-clave-aqui', 'User')
# Reinicia PowerShell para que tenga efecto
```

En Windows CMD:
```cmd
setx OPENAI_API_KEY "tu-clave-aqui"
rem Reinicia CMD
```

En macOS/Linux:
```bash
echo "export OPENAI_API_KEY='tu-clave-aqui'" >> ~/.bashrc
source ~/.bashrc
```

**Opción B: El script solicita interactivamente**
Si no tienes la variable configurada, el script pedirá tu API key cuando se ejecute.

## Ejecución de los ejemplos

### Ejemplo 1: LangChain LLM Chain Básico

```bash
python llm_chain_example.py
```

**Qué sucede:**
1. Se inicializa el modelo GPT-4.1
2. Se crean diferentes templates de prompts
3. Se demuestran cadenas simples con diferentes temas
4. Se muestra encadenamiento múltiple (dos pasos)

**Tiempo esperado:** 30-60 segundos
**Salida:** Respuestas del modelo para 4 ejemplos diferentes

**Ejemplo de salida:**
```
================================================================================
EJEMPLO 1: Pregunta sobre Machine Learning
================================================================================
El aprendizaje supervisado utiliza datos etiquetados...

================================================================================
EJEMPLO 2: Pregunta sobre Python
================================================================================
Una list comprehension es una forma concisa de crear listas...
```

### Ejemplo 2: Sistema RAG Simple (Para Aprendizaje)

```bash
python rag.py
```

**Qué sucede:**
1. Descarga un artículo sobre agentes de IA
2. Divide el contenido en fragmentos indexables
3. Crea un vector store para búsqueda semántica
4. Inicializa un agente con capacidad de recuperación
5. Responde una pregunta usando contexto recuperado

**Tiempo esperado:** 1-2 minutos
**Salida:** Respuesta completa sobre la descomposición de tareas en LLM agents

**Ejemplo de salida:**
```
La descomposición de tareas en agentes LLM implica dividir un objetivo complejo 
en subtareas que el agente puede resolver secuencialmente. Basándome en la 
documentación recuperada...

Source: {url del documento}
Content: {fragmento relevante del documento}
```

### Ejemplo 3: Sistema RAG Avanzado (Para Producción)

```bash
python rag_advanced.py
```

**Características adicionales:**
- Validación de relevancia de documentos
- Logging detallado de operaciones
- Estadísticas de rendimiento
- Configuración centralizada
- Manejo robusto de errores
- Sistema de validación de scores

**Qué sucede:**
1. Configura el sistema con validación
2. Carga documentos y crea índices
3. Procesa múltiples consultas
4. Valida relevancia de resultados (> 0.7)
5. Muestra estadísticas y tiempos

**Tiempo esperado:** 2-3 minutos
**Salida:** Respuestas con scores de relevancia y estadísticas finales

**Ejemplo de salida:**
```
================================================================================
PREGUNTA: ¿Qué es la descomposición de tareas en agentes LLM?
================================================================================

[Relevancia: 0.82]
Fuente: {...}
Contenido: Cuando se enfrenta a tareas complejas...

[Relevancia: 0.78]
Fuente: {...}
Contenido: Los agentes pueden descomponer problemas...

Tiempo de respuesta: 2.34s

================================================================================
ESTADÍSTICAS DEL SISTEMA
================================================================================
{
  "docs_loaded": 1,
  "chunks_created": 15,
  "queries_processed": 3,
  "avg_query_time": 2.12
}
```

## Diferencias entre los Ejemplos

| Aspecto | llm_chain_example.py | rag.py | rag_advanced.py |
|---------|---------------------|--------|-----------------|
| Complejidad | Principiante | Intermedio | Avanzado |
| Caso de uso | Aprendizaje | Desarrollo | Producción |
| Validación | No | No | Sí |
| Logging | No | No | Sí |
| Estadísticas | No | No | Sí |
| Manejo de errores | Básico | Básico | Robusto |
| Configurabilidad | Hardcoded | Hardcoded | Centralizada |
| Líneas de código | ~50 | ~70 | ~300 |

**Recomendación:** 
1. Começar con `llm_chain_example.py` para entender los conceptos
2. Pasar a `rag.py` para ver RAG en acción
3. Estudiar `rag_advanced.py` para técnicas de producción

## Arquitectura del Proyecto

```
RAG Project/
├── llm_chain_example.py      # Ejemplo básico de LLM Chains
├── rag.py                     # Sistema RAG simple
├── rag_advanced.py            # Sistema RAG con validación (producción)
├── requirements.txt           # Dependencias del proyecto
├── .env.example              # Ejemplo de configuración
├── README.md                 # Este archivo
├── DOCUMENTATION.md          # Documentación técnica detallada
├── CONTRIBUTING.md           # Guía para contribuidores
└── .venv/                    # Entorno virtual (creado durante instalación)
```

### Descripción de Archivos

- **llm_chain_example.py**: Demo básico de LangChain mostrando conceptos fundamentales
- **rag.py**: Implementación simple de RAG (para aprendizaje)
- **rag_advanced.py**: Versión de producción con validación, logging y estadísticas
- **requirements.txt**: Lista de dependencias Python
- **.env.example**: Plantilla para variables de entorno
- **README.md**: Documentación principal (este archivo)
- **DOCUMENTATION.md**: Referencia técnica profunda
- **CONTRIBUTING.md**: Guía para desarrolladores que quieran contribuir

### Flujo de Datos

**LLM Chain:**
```
Input Parameters → Prompt Template → LLM Model → Output
                      ↓
                   Variables
                   interpoladas
```

**RAG System:**
```
User Query → Vector Conversion → Similarity Search → Retrieved Docs
              ↓                      ↓
         Embeddings Model      Vector Store
                                              ↓
                                         LLM Agent
                                              ↓
                                    Final Answer with Sources
```

## Personalización

### Cambiar el modelo LLM

En `rag.py` o `llm_chain_example.py`, cambia:

```python
model = init_chat_model("gpt-4.1")  # Cambiar aquí
```

Modelos disponibles:
- `gpt-4.1` (GPT-4 Turbo)
- `gpt-4` (GPT-4)
- `gpt-3.5-turbo` (Más rápido, menos caro)

### Cambiar la fuente de documentos

En `rag.py`, reemplaza la URL:

```python
loader = WebBaseLoader(
    web_paths=("https://nueva-url.com/articulo",),  # Tu URL aquí
    bs_kwargs={"parse_only": bs4_strainer},
)
```

### Ajustar parámetros de división de texto

En `rag.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Aumentar para chunks más grandes
    chunk_overlap=200,      # Mayor solapamiento = más contexto
    add_start_index=True,
)
```

### Cambiar número de documentos recuperados

En `rag.py`, función `retrieve_context`:

```python
retrieved_docs = vector_store.similarity_search(query, k=2)  # 2 es el número de docs
```

## Resolución de Problemas

### Error: "OPENAI_API_KEY not found"

**Solución:**
```bash
# Verifica que la variable de entorno está configurada
echo %OPENAI_API_KEY%  # Windows
echo $OPENAI_API_KEY   # macOS/Linux

# Si no aparece, configúrala como se explicó arriba
```

### Error: "módulo no encontrado" (ImportError)

**Solución:**
```bash
# Verifica que el entorno virtual está activado
# Windows: deberías ver (.venv) en el prompt
# Reinstala las dependencias
pip install -r requirements.txt
```

### Error: "Connection timeout" al descargar documentos

**Solución:**
- Verifica tu conexión a internet
- Intenta con una URL diferente
- Espera unos segundos y vuelve a intentar

### API lenta o timeout

**Solución:**
- Usa un modelo más rápido (gpt-3.5-turbo)
- Reduce el chunk_size
- Aumenta el timeout de la conexión

## Referencias

- [LangChain Official Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart)

- [LangChain Official Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Official Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/)

## Casos de Uso Reales

### LLM Chain - Casos de Aplicación

1. **Generador de Contenido**
   - Blogs y artículos automatizados
   - Social media posts personalizados
   - Descripción de productos

2. **Asistente de Codificación**
   - Generación de código automática
   - Explicación de fragmentos de código
   - Sugerencias y refactoring

3. **Servicio de Traducción**
   - Traducción de contenido multiidoma
   - Localización de aplicaciones
   - Subtítulos generados

### RAG - Casos de Aplicación

1. **Sistemas de Soporte**
   - Help desk automatizado
   - FAQ dinamico basado en documentación
   - Responder tickets de soporte

2. **Análisis Financiero**
   - Análisis de reportes trimestrales
   - Extracción de información de documentos legales
   - Análisis de noticias financieras

3. **Investigación Científica**
   - Búsqueda inteligente de papers
   - Resumen de literatura académica
   - Síntesis de información fragmentada

4. **Sistemas Q&A Especializados**
   - Preguntas sobre documentación interna
   - Bases de conocimiento corporativas
   - Asistentes de dominio específico

## Ventajas del Enfoque RAG

- **Flexibility**: Actualizar la base de conocimiento sin reentrenamiento
- **Transparency**: Las respuestas citan fuentes verificables
- **Efficiency**: No requiere reentrenamiento costoso del modelo
- **Accuracy**: Reduce alucinaciones usando información verificable
- **Scalability**: Funciona con múltiples fuentes de datos heterogéneas
- **Cost-Effective**: Reutiliza modelos pre-entrenados

## Comparativa: LLM Chain vs RAG

| Aspecto | LLM Chain | RAG |
|--------|-----------|-----|
| Complejidad | Simple | Avanzada |
| Casos de uso | Tareas básicas | Tareas complejas basadas en conocimiento |
| Latencia | Baja | Media-Alta |
| Precisión | Moderada | Alta (con fuentes verificables) |
| Dependencia de datos externos | No | Sí |
| Escalabilidad | Alta | Muy alta |
| Alucinaciones | Posibles | Minimizadas |

## Limitaciones Conocidas y Consideraciones

1. **Costo de API**: Cada llamada a OpenAI tiene costo. Los ejemplos usarán tu cuota
2. **Latencia**: Las búsquedas semánticas pueden ser lentas con bases de datos grandes
3. **Vector Store en Memoria**: Los ejemplos usan almacenamiento en memoria (no persistente)
4. **Token Limits**: Existe un límite de tokens por solicitud (~128k para GPT-4 Turbo)
5. **Dependencia de URL**: El ejemplo RAG descarga de una URL específica

## Próximos Pasos y Mejoras

Para expandir este proyecto, considera:

1. **Persistencia**: Usar una base de datos vectorial (Pinecone, Weaviate, Milvus)
2. **Múltiples fuentes**: Integrar PDFs, bases de datos SQL, APIs externas
3. **Caching**: Almacenar embeddings calculados para reutilizar
4. **Validación**: Implementar guardrails y validación de respuestas
5. **Interface**: Crear una UI web con Streamlit o Gradio
6. **Evaluación**: Métodos para evaluar calidad de respuestas (BLEU, ROUGE, etc.)
7. **Fine-tuning**: Entrenar modelos específicos del dominio
8. **Monitoreo**: Logging y trazabilidad de operaciones

## Contribuciones

Este proyecto es educativo. Las contribuciones son bienvenidas:

- Reportar bugs
- Sugerir mejoras
- Agregar más ejemplos
- Mejorar documentación

## Licencia

Este proyecto se proporciona con fines educativos.
