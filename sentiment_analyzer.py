from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from scipy.special import softmax

class HFSentimentAnalyzer:
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval() # Set to evaluation mode
        
    def get_embeddings(self, text):
        """
        Extracts embeddings from the model (CLS token output from the transformer encoder).
        Note: We need to hook into the model or use the base model to get embeddings before the classifier head.
        For 'bert-base-multilingual-uncased-sentiment', the classifier is on top of BERT.
        We can get the output of the BERT part (pooler_output or last_hidden_state).
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            # Get the output from the base BERT model (before the classifier)
            # The model is a SequenceClassification model, so it has a .bert attribute usually.
            # Let's check the model structure or just use the base model.
            # Actually, we can just run the model and get hidden_states if config allows, 
            # or access the base model directly.
            
            # Accessing the base transformer (usually .bert for BertForSequenceClassification)
            if hasattr(self.model, 'bert'):
                outputs = self.model.bert(**inputs)
                # Use pooler_output (CLS token embedding + linear + tanh)
                embeddings = outputs.pooler_output
            elif hasattr(self.model, 'distilbert'):
                 outputs = self.model.distilbert(**inputs)
                 embeddings = outputs.last_hidden_state[:, 0, :] # CLS token
            else:
                # Fallback: try to get hidden states
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1][:, 0, :] # CLS token of last layer
                
        return embeddings.numpy()

    def classify_embeddings(self, embeddings):
        """
        Classifies sentiment from embeddings.
        Args:
            embeddings (np.ndarray): Shape (1, hidden_dim)
        """
        # We need to pass these embeddings through the classifier head.
        # The classifier head is usually self.model.classifier (for BERT) or self.model.pre_classifier + classifier.
        
        # For BertForSequenceClassification:
        # classifier is a Linear layer (dropout + linear)
        
        with torch.no_grad():
            emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
            
            if hasattr(self.model, 'classifier'):
                # Check if it expects dropout
                logits = self.model.classifier(emb_tensor)
            else:
                # Fallback: just run the full model if we can't easily isolate the head? 
                # No, we need to inject embeddings.
                # If we can't easily isolate, this approach is tricky without modifying the model class.
                # Let's assume standard BERT structure for 'nlptown/bert-base-multilingual-uncased-sentiment'
                logits = self.model.classifier(emb_tensor)
                
            probs = softmax(logits.numpy(), axis=1)
            
            # Labels: 1 star to 5 stars
            # 0->1 star, 4->5 stars
            # Map to Negative (0, 1), Neutral (2), Positive (3, 4)
            label_idx = np.argmax(probs)
            
            if label_idx <= 1:
                return 'Negative'
            elif label_idx == 2:
                return 'Neutral'
            else:
                return 'Positive'

    def analyze_sentiment(self, text):
        """
        End-to-end analysis (for True counts).
        """
        emb = self.get_embeddings(text)
        return self.classify_embeddings(emb)

if __name__ == "__main__":
    analyzer = HFSentimentAnalyzer()
    print(analyzer.analyze_sentiment("I love this!"))
    print(analyzer.analyze_sentiment("This is terrible."))
