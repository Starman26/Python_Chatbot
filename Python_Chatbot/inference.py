import os
import tensorflow as tf
import numpy as np
import random
import string
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    """
    Converts text to lowercase and removes punctuation marks.

    """
    return ''.join([char.lower() for char in text if char not in string.punctuation])

def load_tokenizer(filepath):
    """
    Load the tokenizer from a pickle file.
    
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no existe. Asegúrate de haber guardado el tokenizer correctamente.")
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def load_label_encoder(filepath):
    """
    Upload the label encoder from a pickle file.
    
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The document {filepath} does not exist, probably it is not well saved or the path is wrong.")
    with open(filepath, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

def load_responses(json_path):
    """
    Load the JSON file of intentions and build a dictionary
    where each tag has a list of responses associated with it.

    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The document {json_path} does not exist, probably it is not well saved or the path is wrong.")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    responses = {}
    for intent in data.get('intents', []):
        responses[intent['tag']] = intent.get('responses', [])
    return responses

def main():
    # Verificar y cargar el modelo entrenado
    model_path = 'Modeling/chatbot_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at  '{model_path}'. If not working change path with Copy path from Modeling/chatbot_model.h5.")
    model = tf.keras.models.load_model(model_path)
    
    # Cargar el tokenizer y el label encoder previamente guardados
    tokenizer = load_tokenizer('Setup/tokenizer.pickle')
    label_encoder = load_label_encoder('Setup/label_encoder.pickle')
    
    # Cargar el diccionario de respuestas desde el archivo JSON
    responses = load_responses('Dataset/intents.json')
    
    # Obtener la longitud de entrada (maxlen) a partir del modelo.
    # Se asume que el modelo espera entradas con forma (None, input_length)
    input_length = model.input_shape[1]
    
    print("The chatbot is ready. Type 'exit' or 'quit' to exit !! :).")
    
    while True:
        user_input = input(">> ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Preprocesar el input del usuario
        cleaned_input = clean_text(user_input)
        # Convertir el texto a secuencia usando el tokenizer
        seq = tokenizer.texts_to_sequences([cleaned_input])
        # Aplicar padding a la secuencia
        seq = pad_sequences(seq, maxlen=input_length)
        
        # Realizar la predicción
        pred = model.predict(seq)
        pred_class = np.argmax(pred, axis=1)[0]
        tag = label_encoder.inverse_transform([pred_class])[0]
        
        # Seleccionar una respuesta aleatoria basada en el tag predicho
        reply = random.choice(responses.get(tag, ["I don't have an answer for that actually :(."]))
        print("Chatbot:", reply)

if __name__ == '__main__':
    main()
