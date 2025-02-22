import os
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain.chains import ConversationChain  # Cambié ConversationalChain a ConversationChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from huggingface_hub import hf_hub_download
import faiss
import openai

# Cargar variables de entorno
load_dotenv()

# Establecer clave de API para GROQ
groq_api_key = os.getenv("GROQ_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

def leer_saldo_cliente(id_cliente):
    # Leer el archivo CSV
    df = pd.read_csv('saldos.csv')
    
    # Asegurarnos de que id_cliente sea tratado como una cadena (string) para evitar problemas de comparación
    id_cliente = str(id_cliente).strip()  # Convertir id_cliente a string y eliminar espacios
    
    # Buscar el saldo para el ID de cliente dado
    saldo = df.loc[df['id_cliente'].astype(str).str.strip() == id_cliente, 'saldo']
    
    # Verificar si el saldo fue encontrado
    if not saldo.empty:
        return saldo.iloc[0]  # Retorna el saldo encontrado
    else:
        return None  # Si no se encuentra el ID de cliente

# Cargar y procesar las bases de conocimiento
def cargar_base_conocimientos():
    # Cargar los contenidos de los archivos de conocimiento
    with open("knowledge_base/nueva_cuenta.txt", 'r', encoding='utf-8') as f:
        nueva_cuenta = f.read().splitlines()
    
    with open("knowledge_base/tarjeta_credito.txt", 'r', encoding='utf-8') as f:
        tarjeta_credito = f.read().splitlines()

    return nueva_cuenta, tarjeta_credito

# Función para obtener la respuesta usando el modelo de GROQ
openai.api_base = "https://api.groq.com/openai/v1/models"  # URL base correcta de GROQ

def obtener_respuesta_groq(pregunta, fuente):
    try:
        # Hacer la solicitud a la API de GROQ utilizando el cliente compatible con OpenAI
        response = openai.Completion.create(
            model="mixtral-8x7b-32768",  # Nombre del modelo
            prompt=pregunta,
            max_tokens=150
        )
        
        # Obtener la respuesta generada
        respuesta = response.choices[0].text.strip() if response.choices else "No se pudo obtener una respuesta."
        
        return respuesta
    except openai.error.OpenAIError as e:
        # Capturamos cualquier error relacionado con OpenAI/GROQ
        return f"Error al conectar con la API de GROQ: {e}"

# Función para obtener el proceso bancario relacionado con la pregunta
def obtener_proceso_bancario(pregunta, nueva_cuenta, tarjeta_credito):
    # Inicializamos una lista para acumular las líneas que coincidan
    respuesta = []

    # Comprobar si la pregunta contiene información relacionada con "nueva cuenta" o "tarjeta de crédito"
    if "nueva cuenta" in pregunta.lower():
        for linea in nueva_cuenta:
            if any(keyword in linea.lower() for keyword in pregunta.lower().split()):
                respuesta.append(linea)  # Acumular todas las líneas que coincidan
    elif "tarjeta de crédito" in pregunta.lower():
        for linea in tarjeta_credito:
            if any(keyword in linea.lower() for keyword in pregunta.lower().split()):
                respuesta.append(linea)  # Acumular todas las líneas que coincidan
    
    # Si encontramos coincidencias, devolverlas como un solo texto
    if respuesta:
        return "\n".join(respuesta)  # Unir todas las líneas coincidentes en un solo texto
    
    # Si no encontramos coincidencias, usar el modelo de GROQ
    return obtener_respuesta_groq(pregunta, fuente="base_de_conocimientos")

# Función principal para manejar la entrada del cliente
def chatbot_responder(pregunta, nueva_cuenta, tarjeta_credito):
    if "saldo" in pregunta.lower():
        id_cliente = input("Ingrese su ID de cliente: ")
        saldo = leer_saldo_cliente(id_cliente)
        if saldo:
            return f"Su saldo es: {saldo} USD"
        else:
            return "ID de cliente no encontrado."
    else:
        return obtener_proceso_bancario(pregunta, nueva_cuenta, tarjeta_credito)

if __name__ == "__main__":
    # Cargar bases de conocimiento
    nueva_cuenta, tarjeta_credito = cargar_base_conocimientos()

    print("Bienvenido al ChatGroq, ¿cómo puedo ayudarte hoy?")
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() == "salir":
            print("¡Hasta luego!")
            break
        respuesta = chatbot_responder(pregunta, nueva_cuenta, tarjeta_credito)
        print(f"ChatGroq: {respuesta}")