"""
Ejemplo básico de LLM Chain usando LangChain

Este script demuestra los conceptos fundamentales de LangChain:
- Inicialización del modelo de lenguaje
- Creación de prompts
- Encadenamiento de operaciones

Basado en: https://docs.langchain.com/oss/python/langchain/quickstart
"""

import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


# Configuración de API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Introduce tu API key de OpenAI: ")


# Paso 1: Inicializar el modelo de lenguaje
print("Inicializando el modelo LLM...")
model = init_chat_model("gpt-4.1")


# Paso 2: Crear un template de prompt
print("\nCreando template de prompt...\n")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente experto en {tema}. Responde de manera clara y concisa."),
    ("user", "{pregunta}")
])


# Paso 3: Crear la cadena (LLM Chain)
chain = prompt_template | model


# Paso 4: Ejecutar la cadena con diferentes temas
print("=" * 80)
print("EJEMPLO 1: Pregunta sobre Machine Learning")
print("=" * 80)
resultado1 = chain.invoke({
    "tema": "Machine Learning",
    "pregunta": "¿Cuál es la diferencia entre aprendizaje supervisado e insupervisado?"
})
print(resultado1.content)


print("\n" + "=" * 80)
print("EJEMPLO 2: Pregunta sobre Python")
print("=" * 80)
resultado2 = chain.invoke({
    "tema": "Programación en Python",
    "pregunta": "¿Qué es una lista por comprensión (list comprehension)?"
})
print(resultado2.content)


print("\n" + "=" * 80)
print("EJEMPLO 3: Pregunta sobre APIs")
print("=" * 80)
resultado3 = chain.invoke({
    "tema": "Desarrollo de APIs REST",
    "pregunta": "¿Cuál es la diferencia entre POST y PUT en HTTP?"
})
print(resultado3.content)


# Concept adicional: Encadenamiento múltiple
print("\n" + "=" * 80)
print("EJEMPLO 4: Encadenamiento múltiple (cadena de dos pasos)")
print("=" * 80)

# Primer paso: generar un título
prompt_titulo = ChatPromptTemplate.from_messages([
    ("user", "Genera un título creativo para un blog post sobre: {tema}")
])
chain_titulo = prompt_titulo | model

# Segundo paso: generar contenido basado en el título
prompt_contenido = ChatPromptTemplate.from_messages([
    ("user", "Escribe un párrafo de 100 palabras para un blog post con el título: '{titulo}'")
])
chain_contenido = prompt_contenido | model

# Encadenar ambas operaciones
titulo = chain_titulo.invoke({"tema": "Inteligencia Artificial"})
contenido = chain_contenido.invoke({"titulo": titulo.content})

print(f"Título generado: {titulo.content}\n")
print(f"Contenido generado: {contenido.content}")

print("\n" + "=" * 80)
print("FIN DEL EJEMPLO")
print("=" * 80)
