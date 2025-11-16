import numpy as np

def load_character_dict(dict_path):
    """Load character dictionary from a file and prepend 'blank' token."""
    with open(dict_path, "r", encoding="utf-8") as f:
        characters = ["blank"] + [line.strip() for line in f]
    return characters

def postprocess(logits, character_dict_path):
    """
    Post-process CTC-based OCR model output logits into text and confidence score.
    
    Args:
        logits (np.ndarray): Model output of shape [batch, seq_len, num_classes]
        character_dict_path (str): Path to the character dictionary file
    
    Returns:
        tuple: (decoded text, confidence score)
    """
    # Load character dictionary
    characters = load_character_dict(character_dict_path)
    
    # Ensure logits is a numpy array and process the first batch
    if isinstance(logits, list) or isinstance(logits, tuple):
        logits = logits[0]
    preds_idx = np.argmax(logits, axis=2)[0]  # [seq_len]
    preds_prob = np.max(logits, axis=2)[0]    # [seq_len]
    
    # CTC greedy decoding: remove blanks (index 0) and consecutive duplicates
    decoded_indices = []
    decoded_probs = []
    previous = -1  # Initialize with a value not in indices
    
    for i, idx in enumerate(preds_idx):
        if idx != 0 and idx != previous:  # Skip blank and duplicates
            decoded_indices.append(idx)
            decoded_probs.append(preds_prob[i])
        previous = idx
    
    # Map indices to characters, ensuring indices are within dictionary bounds
    text = ''.join([characters[idx] for idx in decoded_indices if idx < len(characters)])
    
    # Compute average confidence score
    confidence = np.mean(decoded_probs) if decoded_probs else 0.0
    
    return text, confidence
