import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_generator import generate_synthetic_data
from sentiment_analyzer import HFSentimentAnalyzer
from dp_utils import clip_and_noise_embeddings

def main():
    print("Starting Privacy-Preserving Sentiment Analysis (Multilingual HF Model)...")
    
    # 1. Generate Data
    print("Generating synthetic data...")
    n_samples = 100 # Reduced samples for BERT performance
    df = generate_synthetic_data(n_samples)
    print(f"Generated {len(df)} samples.")
    
    # Initialize Analyzer
    print("Loading Hugging Face model...")
    analyzer = HFSentimentAnalyzer()
    
    # 2. Analyze Sentiment (True)
    print("Analyzing true sentiment...")
    # We can optimize this by getting embeddings once
    
    true_sentiments = []
    embeddings_list = []
    
    for text in df['text']:
        emb = analyzer.get_embeddings(text)
        embeddings_list.append(emb)
        sent = analyzer.classify_embeddings(emb)
        true_sentiments.append(sent)
        
    df['true_sentiment'] = true_sentiments
    true_counts = df['true_sentiment'].value_counts()
    print("\nTrue Sentiment Counts:")
    print(true_counts)
    
    # 3. Apply Differential Privacy (Input Perturbation on Embeddings)
    epsilon = 1.0 # Privacy budget
    clip_norm = 1.0 # Clipping norm
    print(f"\nApplying Differential Privacy to Embeddings (Epsilon={epsilon}, Clip={clip_norm})...")
    
    # Stack embeddings: (n_samples, hidden_dim)
    # embeddings_list is list of (1, 768) arrays
    all_embeddings = np.vstack(embeddings_list)
    
    # Apply DP
    private_embeddings = clip_and_noise_embeddings(all_embeddings, epsilon, clip_norm)
    
    # 4. Analyze Private Sentiment
    print("Classifying private embeddings...")
    private_sentiments = []
    for i in range(len(private_embeddings)):
        # Reshape to (1, hidden_dim)
        emb = private_embeddings[i].reshape(1, -1)
        sent = analyzer.classify_embeddings(emb)
        private_sentiments.append(sent)
        
    df['private_sentiment'] = private_sentiments
    private_counts = df['private_sentiment'].value_counts()
    print("\nPrivate Sentiment Counts:")
    print(private_counts)
    
    # 5. Visualization
    print("\nGenerating comparison plot...")
    # Ensure all categories are present
    categories = ['Positive', 'Negative', 'Neutral']
    
    true_vals = [true_counts.get(cat, 0) for cat in categories]
    private_vals = [private_counts.get(cat, 0) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, true_vals, width, label='True')
    rects2 = ax.bar(x + width/2, private_vals, width, label='Private')
    
    ax.set_ylabel('Count')
    ax.set_title(f'True vs Private Sentiment (DP on Embeddings, Eps={epsilon})')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    plt.savefig('sentiment_comparison_hf.png')
    print("Plot saved as 'sentiment_comparison_hf.png'.")
    
    # Calculate Accuracy
    accuracy = (df['true_sentiment'] == df['private_sentiment']).mean()
    print(f"\nAgreement (Accuracy) between True and Private: {accuracy:.2f}")
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
