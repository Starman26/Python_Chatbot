from Dataset.data_loader import load_intents_data
from Setup.preprocessing import preprocess_data
from Modeling.model import build_model, train_model, plot_history, save_tokenizer, save_label_encoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import string

def main():
    # As first step on the pipeline, we need to load the data and preprocess it.
    json_path = "Dataset/intents.json"  
    df, responses = load_intents_data(json_path)

    #Train the model
    x_train, y_train, tokenizer, label_encoder, input_shape, vocabulary, output_length = preprocess_data(df)
    model = build_model(input_shape, vocabulary, output_length)
    history = train_model(model, x_train, y_train, epochs=200)
    plot_history(history)
    
    #Save the model as chatbot_model.h5, if you copy the repository you will have the model ready to use
    model.save('Modeling/chatbot_model.h5')
    
    #Same with the tokenizer and label encoder
    save_tokenizer(tokenizer, 'Setup/tokenizer.pickle')
    save_label_encoder(label_encoder, 'Setup/label_encoder.pickle')
    
    # Inference, we will use the model to 
    print("The chatbot is ready. Type 'exit' or 'quit' to exit !! :).")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Clean the input
        cleaned_input = ''.join([char.lower() for char in user_input if char not in string.punctuation])
        
        # Tokenizer and padding
        seq = tokenizer.texts_to_sequences([cleaned_input])
        seq = pad_sequences(seq, maxlen=input_shape)
        
        # Make Predictions
        pred = model.predict(seq)
        pred_class = pred.argmax()
        tag = label_encoder.inverse_transform([pred_class])[0]
        
        # Select a random response for the predicted tag
        print("Chatbot:", random.choice(responses[tag]))
    
if __name__ == "__main__":
    main()
