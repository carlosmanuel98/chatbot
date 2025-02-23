## Propósito de la Aplicación

El objetivo de esta aplicación es proporcionar un chatbot interactivo capaz de responder a preguntas sobre el proceso bancario relacionado con cuentas y tarjetas de crédito, utilizando varias fuentes de conocimiento internas y la API de GROQ para generar respuestas a preguntas más generales.

La aplicación permite al usuario interactuar con el chatbot, realizar consultas sobre el saldo de su cuenta y obtener información detallada sobre productos bancarios como la apertura de nuevas cuentas y las tarjetas de crédito. Para la parte de preguntas generales, se utiliza el modelo de lenguaje de GROQ.

---

## Características Principales

- **Consulta de saldo de cuenta**: El usuario puede consultar el saldo de su cuenta ingresando su ID de cliente.
- **Información sobre nueva cuenta y tarjeta de crédito**: Respuestas generadas a partir de archivos de conocimiento específicos sobre la apertura de nuevas cuentas y tarjetas de crédito.
- **Integración con la API de GROQ**: Para consultas más generales, la aplicación se conecta con la API de GROQ para generar respuestas contextuales basadas en el lenguaje natural.

---

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación utilizado para desarrollar la aplicación.
- **LangChain**: Biblioteca para construir aplicaciones que usan modelos de lenguaje.
- **GROQ API**: API para generar respuestas de lenguaje natural a través de la integración con modelos de lenguaje avanzados.
- **Pandas**: Para la manipulación de datos, específicamente en la carga y manejo de datos de saldo de cliente.
- **FAISS**: Para la búsqueda de información relevante a partir de las bases de conocimiento (aunque actualmente comentado).
- **HuggingFace Embeddings**: Para el procesamiento y generación de representaciones vectoriales (embeddings) de textos.

---

## Requisitos del Proyecto

### Versión de Python

Este proyecto ha sido desarrollado utilizando **Python 3.8**. Asegúrate de tener esta versión o una versión superior de Python instalada en tu entorno para que el proyecto funcione correctamente.

### Dependencias

Las dependencias necesarias para ejecutar esta aplicación están listadas en el archivo `requirements.txt`. Puedes instalar todas las dependencias con el siguiente comando:

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` incluye las siguientes librerías principales:
- `requests`
- `groq`
- `openai`
- `langchain`
- `huggingface_hub`
- `faiss-cpu`
- `pandas`
- `python-dotenv`

---

## Cómo Ejecutar la Aplicación

Para ejecutar la aplicación, sigue estos pasos:

### Configura tu entorno virtual

Si aún no tienes un entorno virtual creado, puedes crear uno con el siguiente comando:

```bash
python -m venv venv_chatbot
```

Luego, activa el entorno virtual:

- **En Windows**:

```bash
venv_chatbot\Scripts\activate
```

- **En MacOS/Linux**:

```bash
source venv_chatbot/bin/activate
```

### Instala las dependencias

Asegúrate de estar en el directorio raíz del proyecto y ejecuta:

```bash
pip install -r requirements.txt
```

### Configura las variables de entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```bash
GROQ_API_KEY=tu_api_key_de_groq
```

### Ejecuta el script principal

Una vez que las dependencias estén instaladas y las variables de entorno configuradas, puedes ejecutar la aplicación con el siguiente comando:

```bash
python chatbot.py
```

### Interactúa con el chatbot

Una vez ejecutado, el chatbot te preguntará cómo puede ayudarte. Puedes realizar preguntas relacionadas con el saldo de cuenta, procesos bancarios de apertura de cuenta y tarjetas de crédito, o preguntas generales que serán respondidas utilizando la API de GROQ.

---

## Estructura de Archivos

El proyecto está organizado de la siguiente manera:

```bash
chatbot/
│
├── chatbot.py          # Script principal de la aplicación
├── requirements.txt    # Archivo con las dependencias necesarias
├── .env                # Archivo para almacenar las variables de entorno (API Keys)
├── knowledge_base/     # Directorio que contiene los archivos de conocimiento (.txt)
│   ├── nueva_cuenta.txt
│   └── tarjeta_credito.txt
└── saldos.csv          # Archivo con los saldos de los clientes
```

---

## Funcionamiento del ChatBot

### Lectura de saldo

Cuando se consulta por el saldo de una cuenta, la aplicación busca el saldo correspondiente en el archivo `saldos.csv`, utilizando el ID del cliente proporcionado.

1. **Recepción del ID de Cliente:**
   - El usuario ingresa su ID de cliente cuando se le solicita.

2. **Búsqueda en el Archivo CSV:**
   - La aplicación carga los datos del archivo `saldos.csv` en un DataFrame de Pandas.
   - Busca el ID del cliente en la columna `id_cliente` del DataFrame.
   - Si encuentra una coincidencia, extrae el saldo correspondiente de la columna `saldo`.

3. **Manejo de Casos No Encontrados:**
   - Si el ID del cliente no se encuentra en el DataFrame, la aplicación devuelve un mensaje indicando que el ID no existe.

### Bases de Conocimiento

La aplicación utiliza dos archivos de texto (`nueva_cuenta.txt` y `tarjeta_credito.txt`) que contienen información sobre los procesos bancarios. Cuando el usuario realiza una pregunta, el ChatBot determina si la pregunta está relacionada con "nueva cuenta" o "tarjeta de crédito" y busca información relevante en el archivo correspondiente.

1. **Determinación de la Categoría de la Pregunta:**
   - El ChatBot analiza la pregunta del usuario para determinar si se relaciona con "nueva cuenta" o "tarjeta de crédito".
   - Si la pregunta no se relaciona con ninguna de estas categorías, utiliza la API de GROQ para generar una respuesta.

2. **Búsqueda de Información Relevante:**
   - Para las preguntas relacionadas con "nueva cuenta", el ChatBot busca palabras clave en el archivo `nueva_cuenta.txt`.
   - Para las preguntas relacionadas con "tarjeta de crédito", el ChatBot busca palabras clave en el archivo `tarjeta_credito.txt`.
   - Compila las líneas que contienen las palabras clave y las devuelve como respuesta.

### Integración con la API de GROQ

Para preguntas que no se encuentran en las bases de conocimiento locales, el ChatBot utiliza la API de GROQ para generar respuestas más generales.

1. **Conexión con la API de GROQ:**
   - La aplicación utiliza la clave API de GROQ almacenada en el archivo `.env` para autenticarse.
   - Envía la pregunta del usuario a la API de GROQ.

2. **Generación de Respuesta:**
   - La API de GROQ procesa la pregunta y genera una respuesta en tiempo real.
   - El ChatBot recibe la respuesta y la muestra al usuario.

3. **Manejo de Errores:**
   - Si ocurre un error al conectarse con la API de GROQ, el ChatBot devuelve un mensaje de error al usuario.

---

## Consideraciones y Futuro

### Escalabilidad

Actualmente, la aplicación maneja una cantidad pequeña de datos. En un escenario de producción, sería importante considerar el uso de bases de datos escalables y sistemas de búsqueda más eficientes.

### Mejoras

Una futura mejora sería implementar un sistema de aprendizaje continuo para adaptar las respuestas del ChatBot con base en las interacciones del usuario.

---

## Diagrama de Flujo

A continuación, se muestra un diagrama de flujo que ilustra cómo el ChatBot procesa las preguntas del usuario:

```
                                      +-------------------+
                                      |   Inicio del     |
                                      |   ChatBot        |
                                      +-------------------+
                                             |
                                             |
                                             v
                                      +-------------------+
                                      | 1. Usuario ingresa  |
                                      |   pregunta          |
                                      +-------------------+
                                             |
                                             |
                                             v
                                      +-------------------+
                                      | 2. Determinar tipo  |
                                      |   de pregunta      |
                                      +-------------------+
                                             |
                                             |
                                             v
                                      +-------------------+---------------------------+
                                      |                                   |
                                      | 3. Pregunta sobre   | 4. Pregunta general   |
                                      |  saldo            |                      |
                                      |                                   |
                                      +-------------------+---------------------------+
                                             |                                       |
                                             |                                       |
                                             v                                        v
                                      +-------------------+          +-------------------+
                                      | 5. Buscar ID en    |          | 6. Conexión con  |
                                      |  saldos.csv       |          |  GROQ API        |
                                      +-------------------+          +-------------------+
                                             |                                       |
                                             |                                       |
                                             v                                        v
                                      +-------------------+          +-------------------+
                                      | 7. Retornar saldo  |          | 8. Generar       |
                                      |  o mensaje de    |          |  respuesta con   |
                                      |  no encontrado   |          |  GROQ           |
                                      +-------------------+          +-------------------+
                                             |                                       |
                                             |                                       |
                                             v                                        v
                                      +-------------------+
                                      | 9. Mostrar        |
                                      |  respuesta al     |
                                      |  usuario          |
                                      +-------------------+
```

Este diagrama ilustra los pasos que sigue el ChatBot para procesar una pregunta, desde la entrada del usuario hasta la generación y muestra de la respuesta.

---

## Conclusión

Esta documentación detalla los componentes, el funcionamiento y las mejoras potenciales del ChatBot integrado con GROQ. La aplicación está diseñada para proporcionar asistencia eficiente a los usuarios en el contexto bancario, utilizando tanto fuentes de conocimiento locales como modelos de lenguaje avanzados. Con las instrucciones y diagramas proporcionados, los desarrolladores pueden comprender y extender la funcionalidad del ChatBot según sea necesario.