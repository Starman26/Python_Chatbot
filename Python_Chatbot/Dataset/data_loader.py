import json
import pandas as pd

def load_intents_data(json_path):
    """
    Load the JSON file with the intentions and returns:
      - A DataFrame with the columns 'patterns' and 'tags'.
      - A 'responses' dictionary where each key is a tag and its value is the list of answers.
      
    Args:
        json_path (str): Path to the JSON file of intentions.
    
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
