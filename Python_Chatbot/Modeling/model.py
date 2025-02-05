from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle

def build_model(input_shape, vocabulary, output_length, embedding_dim=10, lstm_units=10):
    """
    Builds and compiles the model.
    
    Args:
        input_shape (int): Length of the input sequences.
        vocabulary (int): Number of unique words in the vocabulary.
        output_length (int): Number of tags/classes.
        embedding_dim (int): Dimension of the embedding.
        lstm_units (int): Number of units in the LSTM layer.
    
    Returns:
        model: Compiled model.

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
    Train the model.
    
    Args:
        model: Compiled model.
        x_train: Input data.
        y_train: Tags.
        epochs (int): Number of training times.
    
    Returns:
        history: Training history.

    """
    history = model.fit(x_train, y_train, epochs=epochs)
    return history

def plot_history(history):
    """
    Plot accuracy and loss during training.

    """
    plt.plot(history.history['accuracy'], label='Precision of training')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.legend()
    plt.show()

def save_tokenizer(tokenizer, filepath):
    """
   Save the tokenizer in a file.
    
    Args:
        tokenizer: The Keras tokenizer object.
        file path (str): Path where the file will be saved.

    """
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_label_encoder(label_encoder, filepath):
    """
    Save the tag encoder in a pickle file.
    
    Args:
        label_encoder: label encoder object (for example, from sklearn).
        file path (str): Path where the file will be saved.

    """
    with open(filepath, 'wb') as f:
        pickle.dump(label_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
