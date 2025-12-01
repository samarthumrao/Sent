import pandas as pd
import numpy as np
import random

def generate_synthetic_data(n_samples=1000):
    """
    Generates a synthetic dataset of social media posts.
    
    Args:
        n_samples (int): Number of samples to generate.
        
    Returns:
        pd.DataFrame: DataFrame containing 'user_id' and 'text'.
    """
    
    positive_texts = [
        "I love this product! It's amazing.",
        "Had a great day today! #happy",
        "The weather is beautiful.",
        "Just finished a great workout.",
        "So excited for the weekend!",
        "This is the best thing ever.",
        "I am so grateful for my friends.",
        "Delicious food and great company.",
        "Feeling accomplished after finishing my project.",
        "Life is good."
    ]
    
    negative_texts = [
        "I hate this service. Terrible experience.",
        "Feeling down today. #sad",
        "The weather is awful.",
        "Stuck in traffic again. Ugh.",
        "I am so tired and stressed.",
        "This is the worst thing ever.",
        "Disappointed with the results.",
        "Bad food and rude service.",
        "Feeling overwhelmed with work.",
        "Life is hard."
    ]
    
    neutral_texts = [
        "Just woke up.",
        "Going to the store.",
        "Reading a book.",
        "Watching TV.",
        "Eating lunch.",
        "Walking the dog.",
        "Working on my computer.",
        "Listening to music.",
        "Sitting on the couch.",
        "Drinking water."
    ]
    
    data = []
    for i in range(n_samples):
        user_id = f"user_{random.randint(1, n_samples // 2)}" # Simulate some users having multiple posts
        
        category = random.choices(['positive', 'negative', 'neutral'], weights=[0.4, 0.3, 0.3])[0]
        
        if category == 'positive':
            text = random.choice(positive_texts)
        elif category == 'negative':
            text = random.choice(negative_texts)
        else:
            text = random.choice(neutral_texts)
            
        # Add some random variation to text to make it slightly distinct if needed, 
        # but for VADER it doesn't matter much if exact duplicates exist.
        
        data.append({'user_id': user_id, 'text': text})
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_synthetic_data(10)
    print(df.head())
