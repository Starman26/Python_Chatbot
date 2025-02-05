# Chatbot with TensorFlow and LSTM

This project implements a chatbot using an LSTM-based architecture with TensorFlow. The system processes intent data defined in a JSON file, preprocesses it, trains a model, and then interacts with the user in real-time.

## Project Structure

The project is organized into modules to facilitate maintenance and scalability:

- **Dataset/**  
  Contains the data and the logic for loading it.  
  - `intents.json`: File with intents, patterns, and responses.
  - `data_loader.py`: Module for loading and transforming data into a DataFrame and a response dictionary.

- **Setup/**  
  Contains preprocessing scripts.  
  - `preprocessing.py`: Functions for text cleaning, tokenization, padding application, and label encoding.

- **Modeling/**  
  Contains model definition, compilation, training, and evaluation.  
  - `model.py`: Functions to build, train, and visualize the model's performance.

- **pipeline.py**  
  Main script that integrates all stages: data loading, preprocessing, training, and deploying the interactive chatbot.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository

2. **Start the training Pipeline**
   ```bash
    python pipeline.py
   
3. **After training the model, you can initialize the chatbot directly with the following command:**
   ```bash
    python inference.py
