import torch
import os
import glob
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import mean_squared_error, r2_score
import logging
import datetime

# Configuration
NUM_SAMPLES = 30  # Number of samples to test from each category
MODEL_PATH = "best_regressor_run1_epoch_4.pt"  # Update with your actual model path
MODEL_NAME = "roberta-base"  # Model architecture to use
IRRELEVANT_FOLDER = "irrelevant_posts"
RELEVANT_FOLDER = "relevant_posts"
OUTPUT_DIR = "regressor_test_results"
THRESHOLD = 0.5  # Threshold for binary classification from regression scores

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define label constants
IRRELEVANT_LABEL = 0.0  # Posts that are not relevant to the topic
RELEVANT_LABEL = 1.0    # Posts that are relevant to the topic
LABEL_NAMES = {
    IRRELEVANT_LABEL: "irrelevant",
    RELEVANT_LABEL: "relevant"
}

# Define the RegressionHead class (same as in the regressor script)
class RegressionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # Take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def check_model_compatibility(model_path, model_name):
    """
    Check if the saved model is compatible with the current model architecture.
    Returns True if compatible, False if not.
    """
    try:
        # Try to load the state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check if the state dict contains keys specific to the old architecture
        if "distilbert" in str(state_dict.keys()) and "roberta" in model_name:
            logger.warning("Model architecture mismatch detected. The saved model was trained with DistilBERT but you're trying to load it with RoBERTa.")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking model compatibility: {e}")
        return False

def load_model(model_path):
    """
    Load the model and tokenizer from the specified path.
    Handles architecture changes between DistilBERT and RoBERTa.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using {device} for inference")
    
    # Check model compatibility
    if not check_model_compatibility(model_path, MODEL_NAME):
        logger.warning("Attempting to load model with architecture conversion...")
    
    # Load tokenizer
    logger.info(f"Loading model and tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create a new model with regression head
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.classifier = RegressionHead(model.config)
    model.to(device)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Filter out incompatible keys
    filtered_state_dict = {}
    for key, value in state_dict.items():
        # Skip keys that are specific to DistilBERT but not in RoBERTa
        if "roberta.embeddings.position_ids" in key:
            continue
        
        # Map DistilBERT keys to RoBERTa keys if needed
        if "distilbert" in key:
            new_key = key.replace("distilbert", "roberta")
            filtered_state_dict[new_key] = value
        else:
            filtered_state_dict[key] = value
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    logger.info("Model loaded successfully")
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer, device

def predict_single_post(post, model, tokenizer, device, threshold=THRESHOLD):
    """Predict the relevance score of a single post."""
    encoding = tokenizer(
        post,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()
        # Apply sigmoid to get 0-1 range
        score = torch.sigmoid(logits).cpu().numpy()[0]
        # Determine binary classification based on threshold
        is_relevant = score >= threshold
    
    return {
        'score': float(score),
        'is_relevant': bool(is_relevant),
        'threshold': threshold
    }

def get_random_samples(folder_path, num_samples=5):
    """Get random samples from a folder."""
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    selected_files = random.sample(files, min(num_samples, len(files)))
    
    samples = []
    for filepath in selected_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    samples.append({
                        'filename': os.path.basename(filepath),
                        'content': content
                    })
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
    
    return samples

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"regressor_test_results_{timestamp}.txt")
    
    # Load model
    model, tokenizer, device = load_model(MODEL_PATH)
    
    # Get random samples
    irrelevant_samples = get_random_samples(IRRELEVANT_FOLDER, NUM_SAMPLES)
    relevant_samples = get_random_samples(RELEVANT_FOLDER, NUM_SAMPLES)
    
    # Initialize counters for summary
    total_samples = 0
    correct_predictions = 0
    irrelevant_correct = 0
    relevant_correct = 0
    
    # Initialize confusion matrix counters
    true_positives = 0  # Correctly predicted relevant
    false_positives = 0  # Incorrectly predicted as relevant
    true_negatives = 0  # Correctly predicted irrelevant
    false_negatives = 0  # Incorrectly predicted as irrelevant
    
    # Initialize regression metrics
    all_scores = []
    all_true_labels = []
    
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write(f"REGRESSOR MODEL TEST RESULTS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Test irrelevant samples
        f.write("="*80 + "\n")
        f.write(f"TESTING {NUM_SAMPLES} RANDOM IRRELEVANT SAMPLES\n")
        f.write("="*80 + "\n\n")
        
        for sample in irrelevant_samples:
            result = predict_single_post(sample['content'], model, tokenizer, device)
            is_correct = not result['is_relevant']  # For irrelevant samples, prediction should be False
            
            if is_correct:
                irrelevant_correct += 1
                correct_predictions += 1
                true_negatives += 1
            else:
                false_positives += 1
            total_samples += 1
            
            # Store for regression metrics
            all_scores.append(result['score'])
            all_true_labels.append(IRRELEVANT_LABEL)
            
            f.write(f"File: {sample['filename']}\n")
            f.write(f"Relevance Score: {result['score']:.4f}\n")
            f.write(f"Prediction: {LABEL_NAMES[RELEVANT_LABEL if result['is_relevant'] else IRRELEVANT_LABEL]}\n")
            f.write(f"Correct: {'Yes' if is_correct else 'No'}\n")
            
            # Only show content for incorrect predictions
            if not is_correct:
                f.write(f"Text: {sample['content']}\n")
            
            f.write("-"*80 + "\n\n")
        
        # Test relevant samples
        f.write("="*80 + "\n")
        f.write(f"TESTING {NUM_SAMPLES} RANDOM RELEVANT SAMPLES\n")
        f.write("="*80 + "\n\n")
        
        for sample in relevant_samples:
            result = predict_single_post(sample['content'], model, tokenizer, device)
            is_correct = result['is_relevant']  # For relevant samples, prediction should be True
            
            if is_correct:
                relevant_correct += 1
                correct_predictions += 1
                true_positives += 1
            else:
                false_negatives += 1
            total_samples += 1
            
            # Store for regression metrics
            all_scores.append(result['score'])
            all_true_labels.append(RELEVANT_LABEL)
            
            f.write(f"File: {sample['filename']}\n")
            f.write(f"Relevance Score: {result['score']:.4f}\n")
            f.write(f"Prediction: {LABEL_NAMES[RELEVANT_LABEL if result['is_relevant'] else IRRELEVANT_LABEL]}\n")
            f.write(f"Correct: {'Yes' if is_correct else 'No'}\n")
            
            # Only show content for incorrect predictions
            if not is_correct:
                f.write(f"Text: {sample['content']}\n")
            
            f.write("-"*80 + "\n\n")
        
        # Calculate classification metrics
        accuracy = (correct_predictions / total_samples) * 100
        irrelevant_accuracy = (irrelevant_correct / NUM_SAMPLES) * 100
        relevant_accuracy = (relevant_correct / NUM_SAMPLES) * 100
        
        # Calculate precision, recall, and F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate regression metrics
        mse = mean_squared_error(all_true_labels, all_scores)
        r2 = r2_score(all_true_labels, all_scores)
        
        # Write summary
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total samples tested: {total_samples}\n")
        f.write(f"Overall accuracy: {accuracy:.2f}%\n")
        f.write(f"Irrelevant samples accuracy: {irrelevant_accuracy:.2f}% ({irrelevant_correct}/{NUM_SAMPLES})\n")
        f.write(f"Relevant samples accuracy: {relevant_accuracy:.2f}% ({relevant_correct}/{NUM_SAMPLES})\n\n")
        
        f.write("Classification Metrics:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"R-squared (R²): {r2:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"True Positives: {true_positives}\n")
        f.write(f"False Positives: {false_positives}\n")
        f.write(f"True Negatives: {true_negatives}\n")
        f.write(f"False Negatives: {false_negatives}\n")
    
    logger.info(f"Test results saved to {output_file}")
    
    # Print summary to console
    print("\nRegressor Test Results Summary:")
    print(f"Total samples tested: {total_samples}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Irrelevant samples accuracy: {irrelevant_accuracy:.2f}% ({irrelevant_correct}/{NUM_SAMPLES})")
    print(f"Relevant samples accuracy: {relevant_accuracy:.2f}% ({relevant_correct}/{NUM_SAMPLES})")
    print("\nClassification Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nRegression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main() 