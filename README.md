# Chatbot con TensorFlow y LSTM

Este proyecto implementa un chatbot utilizando una arquitectura basada en LSTM con TensorFlow. El sistema procesa datos de intenciones definidas en un archivo JSON, los preprocesa, entrena un modelo y luego interactúa con el usuario en tiempo real.

## Estructura del Proyecto

El proyecto está organizado en módulos para facilitar su mantenimiento y escalabilidad:

- **Dataset/**  
  Contiene los datos y la lógica para cargarlos.  
  - `intents.json`: Archivo con las intenciones, patrones y respuestas.
  - `data_loader.py`: Módulo para cargar y transformar los datos en un DataFrame y un diccionario de respuestas.

- **Setup/**  
  Contiene los scripts de preprocesamiento.  
  - `preprocessing.py`: Funciones para limpiar el texto, tokenizar, aplicar padding y codificar las etiquetas.

- **Modeling/**  
  Contiene la definición, compilación, entrenamiento y evaluación del modelo.  
  - `model.py`: Funciones para construir, entrenar y graficar el rendimiento del modelo.

- **pipeline.py**  
  Archivo principal que integra todas las etapas: carga de datos, preprocesamiento, entrenamiento y puesta en marcha del chatbot interactivo.

## Instalación

1. **Clona el repositorio**

   ```bash
   git clone https://github.com/tu_usuario/tu_repositorio.git
   cd tu_repositorio

2. ** Inicia el codigo **
    ```bash
   python pipeline.py
