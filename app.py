import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentiment_analyzer import HFSentimentAnalyzer
from dp_utils import clip_and_noise_embeddings
import torch

# Set page config
st.set_page_config(page_title="Privacy-Preserving Sentiment Analysis", layout="wide")

@st.cache_resource
def load_analyzer():
    return HFSentimentAnalyzer()

def main():
    st.title("Privacy-Preserving Sentiment Analysis Simulation")
    st.markdown("""
    This application simulates a privacy-preserving sentiment analysis pipeline using **Differential Privacy (DP)**.
    
    **Workflow:**
    1. Upload social media data (Username, Comment).
    2. Extract text embeddings using a Multilingual BERT model.
    3. Apply **DP (Noise)** to the embeddings to hide individual user data.
    4. Classify sentiment from the noisy embeddings.
    """)
    
    # Sidebar for Controls
    st.sidebar.header("Privacy Settings")
    epsilon = st.sidebar.slider("Privacy Budget (Epsilon)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, 
                                help="Lower epsilon means MORE privacy (more noise), but LESS utility (lower accuracy).")
    clip_norm = st.sidebar.slider("Clipping Norm", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                                  help="Maximum L2 norm for embeddings before adding noise.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            # Validate columns
            if 'username' not in df.columns or 'comment' not in df.columns:
                st.error("File must contain 'username' and 'comment' columns.")
                return
                
            st.subheader("1. Raw Data Preview")
            st.dataframe(df.head())
            
            # Load Model
            with st.spinner("Loading AI Model..."):
                analyzer = load_analyzer()
                
            if st.button("Run Analysis"):
                st.subheader("2. Processing & Privacy Barrier")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                true_sentiments = []
                private_sentiments = []
                
                embeddings_list = []
                
                # Process a subset for visualization speed if large
                process_df = df.copy()
                if len(process_df) > 100:
                    st.warning("Processing first 100 rows for speed.")
                    process_df = process_df.head(100)
                
                status_text.text("Extracting Embeddings...")
                
                # Batch processing would be better, but loop for simplicity/progress
                for i, row in process_df.iterrows():
                    text = row['comment']
                    emb = analyzer.get_embeddings(text)
                    embeddings_list.append(emb)
                    
                    # True Sentiment
                    true_sent = analyzer.classify_embeddings(emb)
                    true_sentiments.append(true_sent)
                    
                    progress_bar.progress((i + 1) / len(process_df) * 0.5)
                
                # Convert to numpy
                all_embeddings = np.vstack(embeddings_list)
                
                # Visualize one embedding before/after
                st.markdown("### Visualization of Embedding Perturbation")
                sample_idx = 0
                sample_emb = all_embeddings[sample_idx]
                
                status_text.text("Applying Differential Privacy...")
                
                # Apply DP
                private_embeddings = clip_and_noise_embeddings(all_embeddings, epsilon, clip_norm)
                
                # Visualize Noise
                fig_emb, ax_emb = plt.subplots(1, 2, figsize=(12, 4))
                
                # Plot first 50 dims
                dims = 50
                ax_emb[0].bar(range(dims), sample_emb.flatten()[:dims], color='blue')
                ax_emb[0].set_title("Original Embedding (First 50 dims)")
                ax_emb[0].set_ylim(-2, 2)
                
                ax_emb[1].bar(range(dims), private_embeddings[sample_idx].flatten()[:dims], color='red')
                ax_emb[1].set_title(f"Private Embedding (Epsilon={epsilon})")
                ax_emb[1].set_ylim(-2, 2)
                
                st.pyplot(fig_emb)
                
                status_text.text("Classifying Private Embeddings...")
                
                for i in range(len(private_embeddings)):
                    emb = private_embeddings[i].reshape(1, -1)
                    priv_sent = analyzer.classify_embeddings(emb)
                    private_sentiments.append(priv_sent)
                    progress_bar.progress(0.5 + (i + 1) / len(process_df) * 0.5)
                    
                # ... (previous code) ...
                
                process_df['True Sentiment'] = true_sentiments
                process_df['Private Sentiment'] = private_sentiments
                
                st.subheader("3. Results Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**True Sentiment Counts**")
                    true_counts = process_df['True Sentiment'].value_counts()
                    st.bar_chart(true_counts)
                    
                with col2:
                    st.markdown(f"**Private Sentiment Counts (Epsilon={epsilon})**")
                    private_counts = process_df['Private Sentiment'].value_counts()
                    st.bar_chart(private_counts)
                    
                # Accuracy
                accuracy = (process_df['True Sentiment'] == process_df['Private Sentiment']).mean()
                st.metric("Privacy-Utility Trade-off (Accuracy)", f"{accuracy:.2%}")
                
                st.dataframe(process_df[['username', 'comment', 'True Sentiment', 'Private Sentiment']])
                
                # --- NEW: Detailed Inspection ---
                st.markdown("---")
                st.subheader("4. Detailed Inspection: Before & After DP")
                st.markdown("Select a user to see how their data was privatized.")
                
                selected_user = st.selectbox("Select User", process_df['username'].unique())
                
                if selected_user:
                    user_row = process_df[process_df['username'] == selected_user].iloc[0]
                    idx = process_df.index[process_df['username'] == selected_user][0]
                    
                    st.markdown(f"**Comment:** *{user_row['comment']}*")
                    st.markdown(f"**True Sentiment:** `{user_row['True Sentiment']}` | **Private Sentiment:** `{user_row['Private Sentiment']}`")
                    
                    # Get vectors
                    orig_emb = all_embeddings[idx]
                    priv_emb = private_embeddings[idx]
                    noise_vec = priv_emb - orig_emb
                    
                    # Plotting
                    fig_detail, ax_detail = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                    
                    dims_to_show = 50
                    x_range = range(dims_to_show)
                    
                    # 1. Original
                    ax_detail[0].bar(x_range, orig_emb[:dims_to_show], color='blue', alpha=0.7)
                    ax_detail[0].set_title(f"1. Original Embedding (First {dims_to_show} dims)")
                    ax_detail[0].set_ylabel("Value")
                    
                    # 2. Noise
                    ax_detail[1].bar(x_range, noise_vec[:dims_to_show], color='orange', alpha=0.7)
                    ax_detail[1].set_title(f"2. Added Noise (Laplace, Epsilon={epsilon})")
                    ax_detail[1].set_ylabel("Value")
                    
                    # 3. Private
                    ax_detail[2].bar(x_range, priv_emb[:dims_to_show], color='purple', alpha=0.7)
                    ax_detail[2].set_title(f"3. Private Embedding (Original + Noise)")
                    ax_detail[2].set_ylabel("Value")
                    ax_detail[2].set_xlabel("Dimension Index")
                    
                    plt.tight_layout()
                    st.pyplot(fig_detail)
                    
                    st.info("""
                    **Explanation:**
                    - **Top**: The raw numerical representation of the text (Embedding).
                    - **Middle**: Random noise generated from the Laplace distribution. Lower epsilon = Larger noise bars.
                    - **Bottom**: The final noisy vector used for classification. Notice how it looks similar but 'jittered'.
                    """)

                status_text.text("Simulation Complete.")
                
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
