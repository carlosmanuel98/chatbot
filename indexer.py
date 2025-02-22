import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Función para cargar los archivos de conocimiento
def cargar_base_conocimientos():
    with open("knowledge_base/nueva_cuenta.txt", 'r') as f:
        nueva_cuenta = f.read().splitlines()
    
    with open("knowledge_base/tarjeta_credito.txt", 'r') as f:
        tarjeta_credito = f.read().splitlines()

    return nueva_cuenta, tarjeta_credito

# Función para indexar los conocimientos usando FAISS
def indexar_conocimientos(nueva_cuenta, tarjeta_credito):
    modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Obtener los embeddings de los textos
    embeddings_nueva_cuenta = modelo_embeddings.encode(nueva_cuenta)
    embeddings_tarjeta_credito = modelo_embeddings.encode(tarjeta_credito)
    
    # Crear los índices FAISS
    index_nueva_cuenta = faiss.IndexFlatL2(embeddings_nueva_cuenta.shape[1])
    index_nueva_cuenta.add(embeddings_nueva_cuenta)

    index_tarjeta_credito = faiss.IndexFlatL2(embeddings_tarjeta_credito.shape[1])
    index_tarjeta_credito.add(embeddings_tarjeta_credito)

    return index_nueva_cuenta, index_tarjeta_credito

# Función para guardar los índices en un directorio
def guardar_indices(index_nueva_cuenta, index_tarjeta_credito):
    if not os.path.exists("index"):
        os.makedirs("index")
    
    faiss.write_index(index_nueva_cuenta, "index/nueva_cuenta.index")
    faiss.write_index(index_tarjeta_credito, "index/tarjeta_credito.index")

    print("Índices guardados correctamente en el directorio 'index'.")

# Función principal para ejecutar el proceso de indexación
def reindexar():
    # Cargar los archivos de conocimiento
    nueva_cuenta, tarjeta_credito = cargar_base_conocimientos()

    # Indexar los archivos de conocimiento
    index_nueva_cuenta, index_tarjeta_credito = indexar_conocimientos(nueva_cuenta, tarjeta_credito)

    # Guardar los índices en un directorio 'index'
    guardar_indices(index_nueva_cuenta, index_tarjeta_credito)

if __name__ == "__main__":
    print("Iniciando el proceso de reindexación...")
    reindexar()