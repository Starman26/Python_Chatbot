import json
import pandas as pd

def load_intents_data(json_path):
    """
    Carga el archivo JSON con las intenciones y retorna:
      - Un DataFrame con las columnas 'patterns' y 'tags'.
      - Un diccionario 'responses' donde cada llave es un tag y su valor es la lista de respuestas.
      
    Args:
        json_path (str): Ruta al archivo JSON de intenciones.
    
    Returns:
        tuple: (df, responses)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    inputs = []
    tags = []
    responses = {}
    
    for intent in data.get('intents', []):
        responses[intent['tag']] = intent.get('responses', [])
        for pattern in intent.get('patterns', []):
            inputs.append(pattern)
            tags.append(intent['tag'])
    
    df = pd.DataFrame({'patterns': inputs, 'tags': tags})
    return df, responses
