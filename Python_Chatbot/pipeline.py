from Dataset.data_loader import load_intents_data
from Setup.preprocessing import preprocess_data
from Modeling.model import build_model, train_model, plot_history
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import string

def main():
    # 1. Cargar los datos de intenciones
    json_path = "Dataset/intents.json"  # Asegúrate de que la ruta sea correcta
    df, responses = load_intents_data(json_path)
    
    # 2. Preprocesar los datos (limpieza, tokenización, padding y codificación)
    x_train, y_train, tokenizer, label_encoder, input_shape, vocabulary, output_length = preprocess_data(df)
    
    # 3. Construir el modelo
    model = build_model(input_shape, vocabulary, output_length)
    
    # 4. Entrenar el modelo
    history = train_model(model, x_train, y_train, epochs=200)
    
    # Opcional: graficar la evolución del entrenamiento
    plot_history(history)
    
    # 5. Ciclo interactivo para el chatbot
    print("El chatbot está listo. Escribe 'exit' o 'quit' para salir.")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Limpieza del input del usuario
        cleaned_input = ''.join([char.lower() for char in user_input if char not in string.punctuation])
        
        # Tokenización y padding
        seq = tokenizer.texts_to_sequences([cleaned_input])
        seq = pad_sequences(seq, maxlen=input_shape)
        
        # Predicción
        pred = model.predict(seq)
        pred_class = pred.argmax()
        tag = label_encoder.inverse_transform([pred_class])[0]
        
        # Seleccionar una respuesta aleatoria para el tag predicho
        print("Chatbot:", random.choice(responses[tag]))
    
if __name__ == "__main__":
    main()
