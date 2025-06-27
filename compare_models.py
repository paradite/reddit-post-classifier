import torch
import os
import glob
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import mean_squared_error, r2_score
import logging
import datetime
from typing import Dict, List, Tuple

# Configuration
NUM_SAMPLES = 50  # Number of samples per category for each split
MODEL_NAME = "roberta-base"
IRRELEVANT_FOLDER = "irrelevant_posts"
RELEVANT_FOLDER = "relevant_posts"
OUTPUT_DIR = "model_comparison_results"

# Model paths
CLASSIFIER_MODEL_PATH = "best_model_run12_epoch_9.pt"
REGRESSOR_MODEL_PATH = "best_regressor_run1_epoch_4.pt"
URL_REGRESSOR_MODEL_PATH = "best_url_regressor_run1_epoch_5.pt"
URL_REGRESSOR_MAY_2025_MODEL_PATH = "reddit-url-regressor-may-2025_run1_epoch_6.pt"

# API thresholds for comparison
API_REGRESSOR_THRESHOLD = 0.05
API_URL_REGRESSOR_THRESHOLD = 0.15

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define label constants
IRRELEVANT_LABEL = 0
RELEVANT_LABEL = 1

class RegressionHead(torch.nn.Module):
    """Custom regression head for regressor models"""
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

def load_classifier_model(model_path: str) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """Load the classifier model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    # Load state dict with compatibility handling
    state_dict = torch.load(model_path, map_location=device)
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if "roberta.embeddings.position_ids" in key:
            continue
        if "distilbert" in key:
            new_key = key.replace("distilbert", "roberta")
            filtered_state_dict[new_key] = value
        else:
            filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    return model, tokenizer, device

def load_regressor_model(model_path: str) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """Load a regressor model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.classifier = RegressionHead(model.config)
    model.to(device)
    
    # Load state dict with compatibility handling
    state_dict = torch.load(model_path, map_location=device)
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if "roberta.embeddings.position_ids" in key:
            continue
        if "distilbert" in key:
            new_key = key.replace("distilbert", "roberta")
            filtered_state_dict[new_key] = value
        else:
            filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    return model, tokenizer, device

def predict_classifier(post: str, model, tokenizer, device) -> Dict:
    """Predict using classifier model"""
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

def predict_regressor(post: str, model, tokenizer, device, threshold: float = 0.5) -> Dict:
    """Predict using regressor model"""
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
        score = torch.sigmoid(logits).cpu().numpy()
        
        # Handle both scalar and array outputs
        if isinstance(score, np.ndarray) and score.size > 1:
            score = score[0]
        
        is_relevant = float(score) >= threshold
    
    return {
        'score': float(score),
        'is_relevant': bool(is_relevant),
        'threshold': threshold
    }

def get_random_samples(folder_path: str, num_samples: int = 5) -> List[Dict]:
    """Get random samples from a folder"""
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    selected_files = random.sample(files, min(num_samples, len(files)))
    
    samples = []
    for filepath in selected_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    # Extract URL from first line if present
                    lines = content.split('\n')
                    url = lines[0] if lines[0].startswith('http') else None
                    post_content = '\n'.join(lines[1:]) if url else content
                    
                    samples.append({
                        'filename': os.path.basename(filepath),
                        'content': post_content,
                        'url': url,
                        'full_content': content
                    })
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
    
    return samples

def find_optimal_threshold(scores: List[float], true_labels: List[int]) -> float:
    """Find the optimal threshold that maximizes F1 score"""
    # Use data-driven threshold range based on actual score distribution
    min_score = min(scores)
    max_score = max(scores)
    
    # Create thresholds from slightly below min to slightly above max
    start_thresh = max(0.001, min_score - 0.01)
    end_thresh = min(1.0, max_score + 0.01)
    
    # Use reasonable granularity
    thresholds = np.arange(start_thresh, end_thresh, 0.005)
    
    best_f1 = 0
    best_threshold = (min_score + max_score) / 2  # Default to midpoint
    
    logger.info(f"Searching thresholds from {start_thresh:.4f} to {end_thresh:.4f} (score range: {min_score:.4f} to {max_score:.4f})")
    
    for threshold in thresholds:
        binary_preds = [1 if s >= threshold else 0 for s in scores]
        
        tp = sum(1 for p, t in zip(binary_preds, true_labels) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(binary_preds, true_labels) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(binary_preds, true_labels) if p == 0 and t == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"Best threshold: {best_threshold:.4f} with F1: {best_f1:.4f}")
    return best_threshold

def calculate_metrics(predictions: List[int], true_labels: List[int]) -> Dict:
    """Calculate classification metrics"""
    tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def evaluate_model(model, tokenizer, device, samples: List[Dict], model_type: str, threshold: float = None) -> Dict:
    """Evaluate a single model on the given samples"""
    predictions = []
    scores = []
    true_labels = []
    
    for sample in samples:
        # Determine true label based on folder
        true_label = RELEVANT_LABEL if sample.get('folder', '') == 'relevant' else IRRELEVANT_LABEL
        true_labels.append(true_label)
        
        if model_type == 'classifier':
            result = predict_classifier(sample['content'], model, tokenizer, device)
            predictions.append(result['prediction'])
            scores.append(result['confidence'])
        else:  # regressor
            result = predict_regressor(sample['content'], model, tokenizer, device, threshold or 0.5)
            predictions.append(1 if result['is_relevant'] else 0)
            scores.append(result['score'])
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, true_labels)
    
    # Add regression metrics for regressor models
    if model_type != 'classifier':
        mse = mean_squared_error(true_labels, scores)
        r2 = r2_score(true_labels, scores)
        metrics['mse'] = mse
        metrics['r2'] = r2
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'scores': scores,
        'true_labels': true_labels,
        'threshold': threshold
    }

def main():
    logger.info("Starting model comparison...")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"model_comparison_{timestamp}.txt")
    
    # Get test samples
    logger.info(f"Loading {NUM_SAMPLES*2} samples from each category...")
    irrelevant_samples = get_random_samples(IRRELEVANT_FOLDER, NUM_SAMPLES*2)
    relevant_samples = get_random_samples(RELEVANT_FOLDER, NUM_SAMPLES*2)
    
    # Add folder information to samples
    for sample in irrelevant_samples:
        sample['folder'] = 'irrelevant'
    for sample in relevant_samples:
        sample['folder'] = 'relevant'
    
    # Split data into threshold tuning and evaluation sets (50/50 split)
    tuning_irrelevant = irrelevant_samples[:NUM_SAMPLES]
    tuning_relevant = relevant_samples[:NUM_SAMPLES] 
    tuning_samples = tuning_irrelevant + tuning_relevant
    
    eval_irrelevant = irrelevant_samples[NUM_SAMPLES:]
    eval_relevant = relevant_samples[NUM_SAMPLES:]
    eval_samples = eval_irrelevant + eval_relevant
    
    logger.info(f"Data split: {len(tuning_samples)} samples for threshold tuning, {len(eval_samples)} samples for evaluation")
    
    # Load models
    logger.info("Loading models...")
    classifier_model, classifier_tokenizer, device = load_classifier_model(CLASSIFIER_MODEL_PATH)
    regressor_model, regressor_tokenizer, _ = load_regressor_model(REGRESSOR_MODEL_PATH)
    url_regressor_model, url_regressor_tokenizer, _ = load_regressor_model(URL_REGRESSOR_MODEL_PATH)
    url_regressor_may_2025_model, url_regressor_may_2025_tokenizer, _ = load_regressor_model(URL_REGRESSOR_MAY_2025_MODEL_PATH)
    
    # Find optimal thresholds using tuning data
    logger.info("Finding optimal threshold for regressor model using tuning data...")
    regressor_scores = []
    true_labels = []
    for sample in tuning_samples:
        true_label = RELEVANT_LABEL if sample['folder'] == 'relevant' else IRRELEVANT_LABEL
        true_labels.append(true_label)
        result = predict_regressor(sample['content'], regressor_model, regressor_tokenizer, device, 0.5)
        regressor_scores.append(result['score'])
    
    regressor_threshold = find_optimal_threshold(regressor_scores, true_labels)
    
    logger.info("Finding optimal threshold for URL regressor model using tuning data...")
    url_regressor_scores = []
    for sample in tuning_samples:
        result = predict_regressor(sample['content'], url_regressor_model, url_regressor_tokenizer, device, 0.5)
        url_regressor_scores.append(result['score'])
    
    url_regressor_threshold = find_optimal_threshold(url_regressor_scores, true_labels)
    
    logger.info("Finding optimal threshold for URL regressor May 2025 model using tuning data...")
    url_regressor_may_2025_scores = []
    for sample in tuning_samples:
        result = predict_regressor(sample['content'], url_regressor_may_2025_model, url_regressor_may_2025_tokenizer, device, 0.5)
        url_regressor_may_2025_scores.append(result['score'])
    
    url_regressor_may_2025_threshold = find_optimal_threshold(url_regressor_may_2025_scores, true_labels)
    
    # Evaluate all models on separate evaluation data
    logger.info("Evaluating classifier model on evaluation data...")
    classifier_results = evaluate_model(classifier_model, classifier_tokenizer, device, eval_samples, 'classifier')
    
    logger.info("Evaluating regressor model with optimal threshold on evaluation data...")
    regressor_results = evaluate_model(regressor_model, regressor_tokenizer, device, eval_samples, 'regressor', regressor_threshold)
    
    logger.info("Evaluating regressor model with API threshold on evaluation data...")
    regressor_api_results = evaluate_model(regressor_model, regressor_tokenizer, device, eval_samples, 'regressor', API_REGRESSOR_THRESHOLD)
    
    logger.info("Evaluating URL regressor model with optimal threshold on evaluation data...")
    url_regressor_results = evaluate_model(url_regressor_model, url_regressor_tokenizer, device, eval_samples, 'url_regressor', url_regressor_threshold)
    
    logger.info("Evaluating URL regressor model with API threshold on evaluation data...")
    url_regressor_api_results = evaluate_model(url_regressor_model, url_regressor_tokenizer, device, eval_samples, 'url_regressor', API_URL_REGRESSOR_THRESHOLD)
    
    logger.info("Evaluating URL regressor May 2025 model with optimal threshold on evaluation data...")
    url_regressor_may_2025_results = evaluate_model(url_regressor_may_2025_model, url_regressor_may_2025_tokenizer, device, eval_samples, 'url_regressor', url_regressor_may_2025_threshold)
    
    logger.info("Evaluating URL regressor May 2025 model with API threshold on evaluation data...")
    url_regressor_may_2025_api_results = evaluate_model(url_regressor_may_2025_model, url_regressor_may_2025_tokenizer, device, eval_samples, 'url_regressor', API_URL_REGRESSOR_THRESHOLD)
    
    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL COMPARISON RESULTS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Configuration:\n")
        f.write(f"- Threshold tuning samples: {len(tuning_samples)} ({len(tuning_irrelevant)} irrelevant, {len(tuning_relevant)} relevant)\n")
        f.write(f"- Evaluation samples: {len(eval_samples)} ({len(eval_irrelevant)} irrelevant, {len(eval_relevant)} relevant)\n")
        f.write(f"- Data split: 50% for threshold optimization, 50% for unbiased evaluation\n\n")
        
        f.write("Threshold Analysis:\n")
        f.write(f"- Optimal regressor threshold: {regressor_threshold:.4f}\n")
        f.write(f"- Optimal URL regressor threshold: {url_regressor_threshold:.4f}\n")
        f.write(f"- Optimal URL regressor May 2025 threshold: {url_regressor_may_2025_threshold:.4f}\n")
        f.write(f"- API regressor threshold: {API_REGRESSOR_THRESHOLD}\n")
        f.write(f"- API URL regressor threshold: {API_URL_REGRESSOR_THRESHOLD}\n\n")
        
        f.write("="*80 + "\n")
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        # All model results
        models = [
            ("Classifier", classifier_results),
            ("Regressor (Optimal)", regressor_results),
            ("Regressor (API)", regressor_api_results),
            ("URL Regressor (Optimal)", url_regressor_results),
            ("URL Regressor (API)", url_regressor_api_results),
            ("URL Regressor May 2025 (Optimal)", url_regressor_may_2025_results),
            ("URL Regressor May 2025 (API)", url_regressor_may_2025_api_results)
        ]
        
        for i, (name, results) in enumerate(models, 1):
            f.write(f"{i}. {name.upper()}\n")
            f.write("-" * 40 + "\n")
            metrics = results['metrics']
            
            if results.get('threshold') is not None:
                f.write(f"Threshold: {results['threshold']:.4f}\n")
            
            f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            
            if 'mse' in metrics:
                f.write(f"MSE: {metrics['mse']:.4f}\n")
                f.write(f"R²: {metrics['r2']:.4f}\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"  True Positives (TP): {metrics['tp']} - Correctly predicted as relevant\n")
            f.write(f"  False Positives (FP): {metrics['fp']} - Incorrectly predicted as relevant (actually irrelevant)\n")
            f.write(f"  True Negatives (TN): {metrics['tn']} - Correctly predicted as irrelevant\n")
            f.write(f"  False Negatives (FN): {metrics['fn']} - Incorrectly predicted as irrelevant (actually relevant)\n\n")
        
        # Summary comparison
        f.write("="*80 + "\n")
        f.write("SUMMARY COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}\n")
        f.write("-" * 75 + "\n")
        for name, results in models:
            metrics = results['metrics']
            f.write(f"{name:<25} {metrics['accuracy']:.4f}    {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}\n")
        
        # Best performing model
        f.write("\n" + "="*80 + "\n")
        f.write("BEST PERFORMING MODEL\n")
        f.write("="*80 + "\n\n")
        
        best_f1 = max(results['metrics']['f1'] for _, results in models)
        best_model = next(name for name, results in models if results['metrics']['f1'] == best_f1)
        f.write(f"Best Model: {best_model}\n")
        f.write(f"Best F1 Score: {best_f1:.4f}\n")
    
    # Print summary to console
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<35} {'Accuracy':<10} {'F1 Score':<10} {'Threshold':<10}")
    print("-" * 60)
    for name, results in models:
        threshold_str = f"{results['threshold']:.4f}" if results.get('threshold') is not None else "N/A"
        print(f"{name:<35} {results['metrics']['accuracy']:.4f}    {results['metrics']['f1']:.4f}     {threshold_str}")
    
    best_f1 = max(results['metrics']['f1'] for _, results in models)
    best_model = next(name for name, results in models if results['metrics']['f1'] == best_f1)
    print(f"\nBest Model: {best_model} (F1: {best_f1:.4f})")
    
    print(f"\nDetailed results saved to: {output_file}")
    logger.info(f"Model comparison completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()