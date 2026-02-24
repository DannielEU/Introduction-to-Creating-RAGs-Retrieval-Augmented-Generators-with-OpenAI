# Inicio Rápido - Quick Start Guide

Sin experiencia previa? Sigue estos pasos rápidos para ejecutar el proyecto en 5 minutos.

## 1. Requisitos Previos

- Python 3.8+ instalado
- Una API key de OpenAI (obtén gratis credits en https://platform.openai.com/account/billing/overview)

## 2. Instalación (30 segundos)

```bash
# Clonar el proyecto
git clone <url-del-repositorio>
cd RAG

# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows)
.venv\Scripts\activate

# Activar entorno (macOS/Linux)
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 3. Configurar API Key (30 segundos)

**Windows (más fácil):**
```powershell
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'tu-clave-aqui', 'User')
```

**Alternativa (cualquier SO):**
El script pedirá la clave interactivamente cuando se ejecute.

## 4. Ejecutar Ejemplos (2-3 minutos)

```bash
# Ejemplo 1: LLM Chain básico
python llm_chain_example.py

# Ejemplo 2: Sistema RAG
python rag.py

# Ejemplo 3: RAG avanzado (producción)
python rag_advanced.py
```

## 5. Ver Resultados

Deberías ver en tu terminal:
- Preguntas siendo procesadas
- Respuestas del modelo AI
- Tiempo de ejecución

Completado! Ahora explora lo que hizo y modifica las preguntas para experimentar.

## Pasos Siguientes

- Lee [README.md](README.md) para documentación completa
- Explora [DOCUMENTATION.md](DOCUMENTATION.md) para detalles técnicos
- Modifica las preguntas en los scripts para experimentar
- Cambia la URL del documento en `rag.py` para usar tus propias fuentes

## Solución de Problemas Rápida

| Problema | Solución |
|----------|----------|
| "API key not found" | Configura la variable de entorno como se explica arriba |
| "ModuleNotFoundError" | Ejecuta `pip install -r requirements.txt` |
| "Connection timeout" | Verifica tu conexión a internet |
| "No such file" | Asegúrate de estar en la carpeta correcta |

## Costo Aproximado

Cada ejecución usa aproximadamente:
- **llm_chain_example.py**: $0.01
- **rag.py**: $0.02
- **rag_advanced.py**: $0.05

(Con GPT-4.1 y embeddings de OpenAI)

Para usar un modelo más barato:
```python
# En el archivo, cambiar:
model = init_chat_model("gpt-3.5-turbo")  # 10x más barato
```

## Más Ayuda

- Abre un issue en GitHub
- Consulta [DOCUMENTATION.md](DOCUMENTATION.md)
- Revisa [CONTRIBUTING.md](CONTRIBUTING.md)

Buena suerte!
