import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    """
    Converts text to lowercase and removes punctuation marks.
    
    Args:
        text (str): Text to be cleaned.
    
    Returns:
        str: Text is clear.
    """
    return ''.join([char.lower() for char in text if char not in string.punctuation])

def preprocess_data(df, num_words=2000):
    """
    Preprocess the DataFrame: cleans text, tokenizes, applies padding and encodes labels.
    
    Args:
        df (pd.DataFrame): DataFrame with the 'patterns' and 'tags' columns.
        num_words (int): Maximum number of words to be considered in tokenization.
    
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
