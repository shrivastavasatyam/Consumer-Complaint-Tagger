import torch
import torch.nn.functional as F
from config.label_mappings import get_label_mapping

def classify_input(input_text, models, tokenizer, threshold=0.7):
    # Tokenize the input
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    
    results = {}
    
    # Iterate over the models
    for model_name, model, label_type in models:
        # Get the model's outputs
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply softmax to convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get the maximum probability and corresponding class index
        max_prob, max_index = torch.max(probs, dim=-1)
        
        # If the maximum probability is greater than the threshold, return the class
        if max_prob.item() > threshold:
            label_mapping = get_label_mapping(label_type)
            label = label_mapping[max_index.item()]
            results[model_name] = (label, max_prob.item())
        else:
            results[model_name] = ("Sorry, cannot classify the input.", 0)
    
    return results
