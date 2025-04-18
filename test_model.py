import torch
import os
import glob
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import datetime

# Configuration
NUM_SAMPLES = 30  # Number of samples to test from each category
MODEL_PATH = "reddit_topic_classifier_run3.pt"
# MODEL_PATH = "best_model_run3_epoch_9.pt"
IRRELEVANT_FOLDER = "irrelevant_posts"
RELEVANT_FOLDER = "relevant_posts"
OUTPUT_DIR = "model_test_results"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define label constants
IRRELEVANT_LABEL = 0  # Posts that are not relevant to the topic
RELEVANT_LABEL = 1    # Posts that are relevant to the topic
LABEL_NAMES = {
    IRRELEVANT_LABEL: "irrelevant",
    RELEVANT_LABEL: "relevant"
}

def load_model(model_path):
    """Load the trained model from the specified path."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using {device} for inference")
    
    # Load pre-trained model and tokenizer
    model_name = 'distilbert-base-uncased'
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    return model, tokenizer, device

def predict_single_post(post, model, tokenizer, device):
    """Predict the relevance of a single post."""
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
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
        confidence = probabilities[0][prediction].cpu().numpy()
    
    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'is_relevant': bool(prediction == RELEVANT_LABEL)
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
    output_file = os.path.join(OUTPUT_DIR, f"model_test_results_{timestamp}.txt")
    
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
    
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write(f"MODEL TEST RESULTS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Test irrelevant samples
        f.write("="*80 + "\n")
        f.write(f"TESTING {NUM_SAMPLES} RANDOM IRRELEVANT SAMPLES\n")
        f.write("="*80 + "\n\n")
        
        for sample in irrelevant_samples:
            result = predict_single_post(sample['content'], model, tokenizer, device)
            is_correct = result['prediction'] == IRRELEVANT_LABEL
            
            if is_correct:
                irrelevant_correct += 1
                correct_predictions += 1
                true_negatives += 1
            else:
                false_positives += 1
            total_samples += 1
            
            f.write(f"File: {sample['filename']}\n")
            f.write(f"Prediction: {LABEL_NAMES[result['prediction']]} (confidence: {result['confidence']:.4f})\n")
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
            is_correct = result['prediction'] == RELEVANT_LABEL
            
            if is_correct:
                relevant_correct += 1
                correct_predictions += 1
                true_positives += 1
            else:
                false_negatives += 1
            total_samples += 1
            
            f.write(f"File: {sample['filename']}\n")
            f.write(f"Prediction: {LABEL_NAMES[result['prediction']]} (confidence: {result['confidence']:.4f})\n")
            f.write(f"Correct: {'Yes' if is_correct else 'No'}\n")
            
            # Only show content for incorrect predictions
            if not is_correct:
                f.write(f"Text: {sample['content']}\n")
            
            f.write("-"*80 + "\n\n")
        
        # Calculate metrics
        accuracy = (correct_predictions / total_samples) * 100
        irrelevant_accuracy = (irrelevant_correct / NUM_SAMPLES) * 100
        relevant_accuracy = (relevant_correct / NUM_SAMPLES) * 100
        
        # Calculate precision, recall, and F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Write summary
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total samples tested: {total_samples}\n")
        f.write(f"Overall accuracy: {accuracy:.2f}%\n")
        f.write(f"Irrelevant samples accuracy: {irrelevant_accuracy:.2f}% ({irrelevant_correct}/{NUM_SAMPLES})\n")
        f.write(f"Relevant samples accuracy: {relevant_accuracy:.2f}% ({relevant_correct}/{NUM_SAMPLES})\n\n")
        
        f.write("Detailed Metrics:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"True Positives: {true_positives}\n")
        f.write(f"False Positives: {false_positives}\n")
        f.write(f"True Negatives: {true_negatives}\n")
        f.write(f"False Negatives: {false_negatives}\n")
    
    logger.info(f"Test results saved to {output_file}")
    
    # Print summary to console
    print("\nTest Results Summary:")
    print(f"Total samples tested: {total_samples}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Irrelevant samples accuracy: {irrelevant_accuracy:.2f}% ({irrelevant_correct}/{NUM_SAMPLES})")
    print(f"Relevant samples accuracy: {relevant_accuracy:.2f}% ({relevant_correct}/{NUM_SAMPLES})")
    print("\nDetailed Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main() 