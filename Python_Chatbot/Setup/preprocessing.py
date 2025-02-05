import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    """
    Convierte el texto a minúsculas y elimina signos de puntuación.
    
    Args:
        text (str): Texto a limpiar.
    
    Returns:
        str: Texto limpio.
    """
    return ''.join([char.lower() for char in text if char not in string.punctuation])

def preprocess_data(df, num_words=2000):
    """
    Preprocesa el DataFrame: limpia el texto, tokeniza, aplica padding y codifica las etiquetas.
    
    Args:
        df (pd.DataFrame): DataFrame con las columnas 'patterns' y 'tags'.
        num_words (int): Número máximo de palabras a considerar en la tokenización.
    
    Returns:
        tuple: (x_train, y_train, tokenizer, label_encoder, input_shape, vocabulary, output_length)
    """
    # Limpieza de texto
    df['patterns'] = df['patterns'].apply(clean_text)
    
    # Tokenización
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df['patterns'])
    sequences = tokenizer.texts_to_sequences(df['patterns'])
    
    # Padding
    x_train = pad_sequences(sequences)
    
    # Codificación de etiquetas
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df['tags'])
    
    input_shape = x_train.shape[1]
    vocabulary = len(tokenizer.word_index)
    output_length = len(label_encoder.classes_)
    
    return x_train, y_train, tokenizer, label_encoder, input_shape, vocabulary, output_length
