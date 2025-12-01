import time
import random
import datetime

def yield_tweet():
    """
    Yields a random tweet with a timestamp.
    Simulates a live stream.
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
    
    while True:
        category = random.choices(['positive', 'negative', 'neutral'], weights=[0.4, 0.3, 0.3])[0]
        
        if category == 'positive':
            text = random.choice(positive_texts)
        elif category == 'negative':
            text = random.choice(negative_texts)
        else:
            text = random.choice(neutral_texts)
            
        user_id = f"user_{random.randint(100, 999)}"
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        yield {
            "timestamp": timestamp,
            "username": user_id,
            "comment": text
        }
        
        # Simulate variable delay
        time.sleep(random.uniform(0.5, 1.5))
