import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.special import softmax
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt

# import nltk
# nltk.download('punkt')

def generate_summary(text, model, tokenizer, max_length=256):
    """Generate a concise summary of 1-3 sentences from the input text."""
    system_prompt = "You are a helpful assistant. Please provide a very concise summary of the following text in 1-2 sentences."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    outputs = model.generate(
        inputs, 
        max_new_tokens=max_length,
        temperature=0.3, 
        top_p=0.9,
        do_sample=True,
    )
    
    full_output = tokenizer.decode(outputs[0])
    summary = full_output.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    
    return summary

def load_model():
    
    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    
    return model, tokenizer

def load_embedding_model():
    checkpoint = "BAAI/bge-m3"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = SentenceTransformer(checkpoint).to(device)
    
    return model

def load_cross_encoder_model():
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
    model.eval()
    return model, tokenizer

def calculate_attributions(original_text, summary, model, cross_encoder_tokenizer=None, model_type='embedding', agg='mean', normalization_type=None, window_size=1):
    """
    Calculate attribution scores between summary and input text sentences using either embedding or cross-encoder model.
    
    Args:
        original_text: Input text to analyze
        summary: Generated summary text
        model: Either SentenceTransformer for embeddings or AutoModelForSequenceClassification for cross-encoding
        cross_encoder_tokenizer: Tokenizer for cross-encoder model (only needed if model_type='cross-encoder')
        model_type: Either 'embedding' or 'cross-encoder'
        agg: Aggregation method ('mean', 'max', or 'weighted') - only used for embedding model
        normalization_type: Type of normalization to apply to scores ('softmax', 'min-max', or None)
        window_size: Number of sentences to consider in each window (default=1 for single sentences)
    
    Returns:
        Tuple containing:
        - Dictionary mapping input sentences to their attribution scores
        - List of input sentences
        - List of summary sentences
        - Similarity matrix
    """
    # Split texts into sentences
    input_sentences = sent_tokenize(original_text)
    summary_sentences = sent_tokenize(summary)
    
    # Create sliding windows of sentences
    input_windows = []
    for i in range(len(input_sentences) - window_size + 1):
        window = ' '.join(input_sentences[i:i + window_size])
        input_windows.append(window)
    
    if model_type == 'embedding':
        # Calculate embeddings for windows instead of single sentences
        input_embeddings = model.encode(input_windows)
        summary_embeddings = model.encode(summary_sentences)
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(summary_sentences), len(input_windows)))
        for i, sum_emb in enumerate(summary_embeddings):
            for j, inp_emb in enumerate(input_embeddings):
                similarity = np.dot(sum_emb, inp_emb) / (np.linalg.norm(sum_emb) * np.linalg.norm(inp_emb))
                similarity_matrix[i, j] = similarity
                
    elif model_type == 'cross-encoder':  # cross-encoder
        # Calculate relevance scores using cross-encoder
        similarity_matrix = np.zeros((len(summary_sentences), len(input_windows)))
        
        for i, sum_sent in enumerate(summary_sentences):
            # Create pairs for cross-encoding
            pairs = [[sum_sent, inp_window] for inp_window in input_windows]
            
            # Calculate scores in batches to avoid memory issues
            batch_size = 8
            for batch_start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[batch_start:batch_start + batch_size]
                
                with torch.no_grad():
                    inputs = cross_encoder_tokenizer(batch_pairs, padding=True, truncation=True, 
                                    return_tensors='pt', max_length=512)
                    # Move inputs to same device as model
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    scores = model(**inputs, return_dict=True).logits.view(-1,).float()
                    
                    similarity_matrix[i, batch_start:batch_start + len(batch_pairs)] = scores.cpu().numpy()
    else:
        raise ValueError(f"Invalid model type: {model_type}. Please use 'embedding' or 'cross-encoder'.")
    
    # Expand window scores back to individual sentences
    if window_size > 1:
        # Initialize expanded matrix with zeros
        expanded_scores = np.zeros((similarity_matrix.shape[0], len(input_sentences)))
        
        # For each window, assign its score to all sentences in that window
        for i in range(len(input_windows)):
            for j in range(window_size):
                if i + j < len(input_sentences):
                    expanded_scores[:, i + j] = np.maximum(
                        expanded_scores[:, i + j],
                        similarity_matrix[:, i]
                    )
        similarity_matrix = expanded_scores
    
    # Apply normalization if specified
    if normalization_type == 'softmax':
        scores = softmax(similarity_matrix, axis=1)
    elif normalization_type == 'min-max':
        # Min-max normalization as alternative
        scores = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
    else:
        scores = similarity_matrix

    # For cross-encoder, we typically just use the raw scores
    if model_type == 'cross-encoder':
        final_scores = np.mean(scores, axis=0)  # Average across summary sentences
    else:
        # Aggregate scores based on specified method (only for embedding model)
        if agg == 'max':
            final_scores = np.max(scores, axis=0)
        elif agg == 'weighted':
            weights = softmax(np.max(similarity_matrix, axis=1))
            final_scores = np.average(scores, axis=0, weights=weights)
        else:  # default to mean
            final_scores = np.mean(scores, axis=0)
    
    # Create attribution dictionary
    attributions = {
        sentence: score 
        for sentence, score in zip(input_sentences, final_scores)
    }
    
    return attributions, input_sentences, summary_sentences, scores

def create_token_heatmap(sentences, attributions, line_height=5.0, show=False):
    # Normalize attributions
    attributions = np.array(attributions, dtype=float)
    min_attr = np.min(attributions)
    max_attr = np.max(attributions)
    norm_attr = (attributions - min_attr) / (max_attr - min_attr)
    
    # Create figure with more height
    total_height = len(sentences) * line_height
    fig = plt.figure(figsize=(12, min(12, max(6, total_height/2))), dpi=100)  # Added min() to cap height, reduced scaling
    gs = fig.add_gridspec(1, 2, width_ratios=[30, 1])
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])
    
    # Remove axis decorations
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    current_y = total_height - line_height
    renderer = fig.canvas.get_renderer()
    
    # Initialize with first word of first sentence
    if sentences and len(sentences[0].split()) > 0:
        color = plt.cm.viridis(norm_attr[0])
        text = ax.text(0.1, current_y, sentences[0].split()[0], 
                      color=color,
                      fontsize=12)
    
    # Process all sentences continuously
    for i, (sentence, score) in enumerate(zip(sentences, norm_attr)):
        color = plt.cm.viridis(score)
        words = sentence.split()
        
        # Skip the first word of first sentence as it's already placed
        start_idx = 1 if i == 0 else 0
        
        for word in words[start_idx:]:
            # Get the current text's bbox
            prev_bbox = text.get_window_extent(renderer=renderer)
            
            text = ax.annotate(
                f" {word}", 
                xycoords=text, 
                xy=(1, 0),
                xytext=(2, 0),  # Small horizontal gap between words
                textcoords="offset points",
                color=color,
                fontsize=12,
                verticalalignment="bottom",
            )
            
            # Check if we need to wrap to next line
            bbox = text.get_window_extent(renderer=renderer)
            if bbox.x1 > ax.get_window_extent(renderer=renderer).x1 * 1.0:
                current_y -= line_height
                text = ax.text(0.1, current_y, word,
                            color=color,
                            fontsize=12,
                           )
    
    # Adjust limits
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-line_height, total_height + line_height)
    
    # Add colorbar
    norm = plt.Normalize(vmin=min_attr, vmax=max_attr)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    plt.colorbar(sm, cax=cax, label='Attribution Score')
    
    plt.suptitle('Text Attribution Visualization (Lighter = Higher Attribution)', fontsize=12)
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def determine_threshold(similarity_matrix, method='dynamic'):
    """
    Automatically determine similarity threshold using various methods.
    
    Methods:
    - 'percentile': Use statistics of the similarity distribution
    - 'elbow': Find elbow point in sorted similarities
    - 'dynamic': Adaptive thresholding based on distribution gaps
    """
    # Flatten similarity matrix and remove zeros/very low values
    similarities = similarity_matrix.flatten()
    similarities = similarities[similarities > 0.1]  # Remove noise floor
    
    if method == 'percentile':
        # Use mean + 1 std dev as threshold
        return float(np.mean(similarities) + np.std(similarities))
    
    elif method == 'elbow':
        # Sort similarities and find "elbow" point
        sorted_sims = np.sort(similarities)
        n_points = len(sorted_sims)
        
        # Calculate curvature at each point
        max_curvature = 0
        threshold = 0.3  # fallback
        
        for i in range(1, n_points - 1):
            # Approximate curvature using three points
            y_diff = sorted_sims[i+1] - 2*sorted_sims[i] + sorted_sims[i-1]
            x_diff = 1  # constant x-spacing
            curvature = abs(y_diff / (1 + x_diff**2)**1.5)
            
            if curvature > max_curvature:
                max_curvature = curvature
                threshold = sorted_sims[i]
        
        return float(threshold)
    
    else:  # method == 'dynamic'
        # Find natural breaks in the similarity distribution
        sorted_sims = np.sort(similarities)
        gaps = sorted_sims[1:] - sorted_sims[:-1]
        
        # Calculate local statistics in windows
        window_size = max(len(gaps) // 10, 1)  # 10% of data points
        significance_factors = []
        
        for i in range(len(gaps) - window_size):
            window = gaps[i:i+window_size]
            local_mean = np.mean(window)
            local_std = np.std(window)
            if local_std == 0:
                significance_factors.append(0)
            else:
                # How many standard deviations is this gap from the local mean?
                significance_factors.append((gaps[i] - local_mean) / local_std)
        
        # Find the most significant gap
        if significance_factors:
            max_significance_idx = np.argmax(significance_factors)
            threshold = sorted_sims[max_significance_idx]
            
            # Ensure threshold is reasonable
            min_threshold = np.percentile(similarities, 60)  # At least top 40%
            max_threshold = np.percentile(similarities, 90)  # At most top 10%
            threshold = np.clip(threshold, min_threshold, max_threshold)
        else:
            threshold = np.percentile(similarities, 75)  # fallback to top 25%
            
        return float(threshold)

def plot_sankey_diagram(similarity_matrix, input_sentences, summary_sentences, max_connections=3):
    """
    Create a Sankey diagram using pySankey library with fixed order of sentences.
    """
    from pysankey import sankey
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Automatically determine threshold
    threshold = determine_threshold(similarity_matrix, method='dynamic')
    
    def truncate_text(text, max_len=50):
        return text[:max_len] + "..." if len(text) > max_len else text
    
    # Create input and summary labels with truncated text
    input_label_map = {
        f'Input {i+1}': f'Input {i+1}: {truncate_text(sent)}' 
        for i, sent in enumerate(input_sentences)
    }
    summary_label_map = {
        f'Summary {i+1}': f'Summary {i+1}: {truncate_text(sent)}' 
        for i, sent in enumerate(summary_sentences)
    }
    
    # Create lists for the diagram
    lefts = []  # Input sentence indices
    rights = []  # Summary sentence indices
    weights = []  # Similarity scores
    
    # Filter connections and create data
    for i in range(len(summary_sentences)):
        scores = similarity_matrix[i]
        top_indices = np.argsort(scores)[-max_connections:]
        top_scores = scores[top_indices]
        
        # Only keep connections above threshold
        valid_mask = top_scores > threshold
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]
        
        for idx, score in zip(top_indices, top_scores):
            lefts.append(f'Input {idx+1}')
            rights.append(f'Summary {i+1}')
            weights.append(float(score))
    
    # Get unique labels in reverse order
    unique_lefts = sorted(set(lefts), key=lambda x: -int(x.split()[1]))
    unique_rights = sorted(set(rights), key=lambda x: -int(x.split()[1]))
    
    # Create the label lists with full text
    leftLabels = [input_label_map[label] for label in unique_lefts]
    rightLabels = [summary_label_map[label] for label in unique_rights]
    
    # Update the actual connection labels to include the text
    lefts = [input_label_map[label] for label in lefts]
    rights = [summary_label_map[label] for label in rights]
    
    # Create color dictionary
    colors = plt.cm.Set2(np.linspace(0, 1, len(summary_sentences)))
    colorDict = {}
    
    # Add colors for all input labels (using a neutral color)
    for label in leftLabels:
        colorDict[label] = mcolors.rgb2hex(plt.cm.Greys(0.2))
    
    # Add colors for summary labels (using distinct colors)
    for i, label in enumerate(rightLabels):
        colorDict[label] = mcolors.rgb2hex(colors[i])
    
    # Create figure
    plt.figure(figsize=(15, max(8, len(input_sentences) * 0.5)))
    
    # Create Sankey diagram
    sankey(
        left=lefts,
        right=rights,
        leftWeight=weights,
        rightWeight=weights,
        colorDict=colorDict,
        leftLabels=leftLabels,
        rightLabels=rightLabels,
        aspect=20,
        fontsize=8,
        rightColor=True  # Color based on right (summary) labels
    )
    
    plt.title(f'Information Flow (threshold={threshold:.3f}, max_connections={max_connections})')
    plt.tight_layout()
    plt.show()

def run_parameter_sweep(text, summary, embedding_model, cross_encoder_model, cross_encoder_tokenizer):
    """Run parameter sweep across model types, window sizes, and aggregation methods."""
    from pathlib import Path
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Parameter combinations to test
    model_types = ['embedding', 'cross-encoder']
    window_sizes = [1, 2, 3]
    agg_methods = ['mean', 'max']  # New parameter
    
    for model_type in model_types:
        # Select appropriate model and tokenizer
        if model_type == 'embedding':
            model = embedding_model
            tokenizer = None
        else:  # cross-encoder case
            model = cross_encoder_model
            tokenizer = cross_encoder_tokenizer
        # Use all aggregation methods for both model types
        aggs_to_test = agg_methods
            
        for window_size in window_sizes:
            for agg in aggs_to_test:
                # Calculate attributions
                attributions, input_sentences, summary_sentences, similarity_matrix = calculate_attributions(
                    text, 
                    summary, 
                    model,
                    cross_encoder_tokenizer=tokenizer,
                    model_type=model_type,
                    window_size=window_size,
                    agg=agg  # Add aggregation parameter
                )
                
                # Create and save heatmap
                fig = create_token_heatmap(input_sentences, list(attributions.values()), show=False)
                filename = f"plots/heatmap_{model_type}_window{window_size}_agg{agg}.png"
                fig.savefig(filename, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Saved {filename}")

def main():
    text = """This section briefly summarizes the state of the art in the area of semantic segmentation and se-
mantic instance segmentation. As the majority of state-of-the-art techniques in this area are deep
learning approaches we will focus on this area. Early deep learning-based approaches that aim at
assigning semantic classes to the pixels of an image are based on patch classification. Here the
image is decomposed into superpixels in a preprocessing step e.g. by applying the SLIC algorithm
[1]. The superpixels are then padded and classified by using a neural network architecture for im-
age classification. Typically, nowadays there are Convolutional Neural Networks which consist of
a series of 2D convolutions, followed by a number of fully connected layers that use the extracted
features predict a probability for each possible class 2.1.

Other approaches are based on so-called Fully Convolutional Neural Networks (FCNs). Here
not an image patch but the whole image are taken as input and the output is a two-dimensional
feature map that assigns class probabilities to each pixel. Conceptually FCNs are similar to CNNs
used for classification but the fully connected layers are usually replaced by transposed convolu-
tions which have learnable parameters and can learn to upsample the extracted features to the final
pixel-wise classification result.

Standard architectures FCN architectures that are commonly used for semantic segmentation are
e.g. U-Net [73] or architectures with VGG [82] or ResNet-based [34] feature encoders. The archi-
tecture change leads to several advantages such as better computational efficiency, less parameters
and that the network can process images of varying size.

As explained before in semantic segmentation the network only predicts semantic labels for
each pixel of an image. This can be sufficient to also identify object instances in a post-processing
step but as there usually are also challenging cases with e.g. overlapping instances there is also
the need for network architectures that additionally assign instance labels to the input pixels. In
the following two types of deep learning-based semantic instance segmentation techniques will be
reviewed – proposal-based and instance embedding-based techniques."""

    model, tokenizer = load_model()
    embedding_model = load_embedding_model()
    cross_encoder_model, cross_encoder_tokenizer = load_cross_encoder_model()
    summary = generate_summary(text, model, tokenizer)

    print("\nSummary:")
    print(summary)
    
    # Run parameter sweep and save plots
    run_parameter_sweep(
        text, 
        summary, 
        embedding_model, 
        cross_encoder_model, 
        cross_encoder_tokenizer
    )

    # Plot Sankey diagram
    attributions, input_sentences, summary_sentences, similarity_matrix = calculate_attributions(text, summary, embedding_model, model_type='embedding', window_size=1, agg='mean')
    plot_sankey_diagram(similarity_matrix, input_sentences, summary_sentences)


if __name__ == "__main__":
    main()