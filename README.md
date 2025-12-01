# Privacy-Preserving Sentiment Analysis (Multilingual)

This project implements a privacy-preserving sentiment analysis system using Differential Privacy (DP) on text embeddings. It uses a multilingual Hugging Face model (`nlptown/bert-base-multilingual-uncased-sentiment`) to analyze sentiment while ensuring the input data (embeddings) is privatized before classification.

## Features
- **Multilingual Support**: Uses a BERT-based model supporting English, Spanish, German, French, etc.
- **Input Perturbation**: Applies Differential Privacy (Clipping + Laplace Noise) directly to the text embeddings.
- **Frontend Simulation**: A Streamlit app to visualize the process with custom data.
- **Synthetic Data**: Generates synthetic social media posts for demonstration.
- **Visualization**: Compares true sentiment counts with differentially private counts.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Command Line Interface
Run the main script to generate synthetic data and see the analysis:
```bash
python main.py
```

### 2. Frontend Simulation (Streamlit)
Run the interactive web app to upload your own data (CSV/Excel):
```bash
streamlit run app.py
```
Or if that fails:
```bash
python -m streamlit run app.py
```
- Upload a file with `username` and `comment` columns (see `sample_data.csv`).
- Adjust the **Privacy Budget (Epsilon)** to see how noise affects the results.

## Files
- `main.py`: CLI orchestrator.
- `app.py`: Streamlit web application.
- `data_generator.py`: Generates synthetic data.
- `sentiment_analyzer.py`: Hugging Face model wrapper.
- `dp_utils.py`: Implements DP for embeddings.
