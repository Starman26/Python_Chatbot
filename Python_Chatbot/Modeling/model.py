from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

def build_model(input_shape, vocabulary, output_length, embedding_dim=10, lstm_units=10):
    """
    Construye y compila el modelo.
    
    Args:
        input_shape (int): Longitud de las secuencias de entrada.
        vocabulary (int): Número de palabras únicas en el vocabulario.
        output_length (int): Número de etiquetas/clases.
        embedding_dim (int): Dimensión del embedding.
        lstm_units (int): Número de unidades en la capa LSTM.
    
    Returns:
        model: Modelo compilado.
    """
    i = Input(shape=(input_shape,))
    x = Embedding(vocabulary + 1, embedding_dim)(i)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(output_length, activation="softmax")(x)
    
    model = Model(inputs=i, outputs=x)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=200):
    """
    Entrena el modelo.
    
    Args:
        model: Modelo compilado.
        x_train: Datos de entrada.
        y_train: Etiquetas.
        epochs (int): Número de épocas de entrenamiento.
    
    Returns:
        history: Historial de entrenamiento.
    """
    history = model.fit(x_train, y_train, epochs=epochs)
    return history

def plot_history(history):
    """
    Grafica la precisión y la pérdida durante el entrenamiento.
    """
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.legend()
    plt.show()
