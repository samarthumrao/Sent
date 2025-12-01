import numpy as np

def laplace_mechanism(value, sensitivity, epsilon):
    """
    Adds Laplacian noise to a value for Differential Privacy.
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

def get_private_count(count, epsilon):
    """
    Returns a differentially private count.
    """
    return laplace_mechanism(count, 1.0, epsilon)

def clip_and_noise_embeddings(embeddings, epsilon, clip_norm=1.0):
    """
    Applies Differential Privacy to embeddings using the Laplace Mechanism.
    
    Args:
        embeddings (np.ndarray): The embeddings to privatize. Shape (batch_size, hidden_dim).
        epsilon (float): The privacy budget.
        clip_norm (float): The maximum L2 norm for clipping.
        
    Returns:
        np.ndarray: The noisy embeddings.
    """
    # 1. Clipping
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    scaling_factors = np.minimum(1, clip_norm / (norms + 1e-10)) # Avoid division by zero
    clipped_embeddings = embeddings * scaling_factors
    
    # 2. Noise Addition
    # Sensitivity is 2 * clip_norm for L1 mechanism on vectors? 
    # Actually, for vector mechanism, if we use Laplace noise on each coordinate:
    # Sensitivity of each coordinate is bounded by clip_norm (L2 norm <= C => L1 norm <= sqrt(d)*C)
    # But usually for "d-dimensional" vector, we might use Gaussian mechanism for L2 sensitivity.
    # Here, let's stick to a simple coordinate-wise Laplace for demonstration, 
    # assuming L1 sensitivity is bounded by clip_norm * sqrt(dim) or similar, 
    # OR we just treat sensitivity as clip_norm for simplicity in this demo context.
    # A more rigorous approach would use Gaussian mechanism with L2 sensitivity = clip_norm.
    
    # Let's use Laplace noise on each coordinate.
    # If L2 norm is clipped to C, then L1 norm is at most C * sqrt(d).
    # Sensitivity = C * sqrt(d).
    
    dim = embeddings.shape[1]
    sensitivity = clip_norm * np.sqrt(dim) # Conservative bound for L1 sensitivity given L2 clip
    
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=embeddings.shape)
    
    return clipped_embeddings + noise
