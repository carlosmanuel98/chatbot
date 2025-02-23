import os
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain.chains import ConversationChain  # Cambié ConversationalChain a ConversationChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from huggingface_hub import hf_hub_download
import faiss
from groq import Groq  # Importamos la librería de GROQ
# Cargar variables de entorno
load_dotenv()

# Establecer clave de API para GROQ
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),  # Usamos la variable de entorno para la clave de API
)


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
def obtener_respuesta_groq(pregunta, fuente):
    try:
        # Realizamos la solicitud para completar el chat usando el modelo de GROQ
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": pregunta,  # Usamos la pregunta que el usuario hace
                }
            ],
            model="llama-3.3-70b-versatile",  # Modelo específico que deseas usar
        )

        # Retornamos la respuesta del modelo
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error al conectar con la API de GROQ: {str(e)}"

# Ejemplo de uso
respuesta = obtener_respuesta_groq("Explain the importance of fast language models", fuente="general")
print(f"Respuesta de GROQ: {respuesta}")

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
        pregunta = input("\nTú: ")
        if pregunta.lower() == "salir":
            print("¡Hasta luego!")
            break
        respuesta = chatbot_responder(pregunta, nueva_cuenta, tarjeta_credito)
        print(f"ChatGroq: {respuesta}")