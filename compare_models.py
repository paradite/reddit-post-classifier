import torch
import os
import glob
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
import logging
import datetime
from typing import Dict, List, Tuple
from collections import Counter

# Configuration
NUM_SAMPLES = 200  # Larger sample size for more robust evaluation
MODEL_NAME = "roberta-base"
IRRELEVANT_FOLDER = "irrelevant_posts"
RELEVANT_FOLDER = "relevant_posts"
OUTPUT_DIR = "model_comparison_results"
N_FOLDS = 5  # For cross-validation

# Model paths
CLASSIFIER_MODEL_PATH = "best_model_run12_epoch_9.pt"
REGRESSOR_MODEL_PATH = "best_regressor_run1_epoch_4.pt"
URL_REGRESSOR_MODEL_PATH = "best_url_regressor_run1_epoch_5.pt"

# API server thresholds for comparison
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
    """Custom regression head for regressor models - matches API server exactly"""
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):  # Match API server signature
        x = features[:, 0, :]  # Take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def load_classifier_model(model_path: str) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """Load the classifier model - matches API server loading"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    # Load state dict with compatibility handling - matches API server
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
    """Load a regressor model - matches API server loading"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.classifier = RegressionHead(model.config)
    model.to(device)
    
    # Load state dict with compatibility handling - matches API server
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
        'is_relevant': bool(prediction == RELEVANT_LABEL),
        'probabilities': probabilities[0].cpu().numpy().tolist()
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
        'threshold': threshold,
        'raw_logits': float(logits.cpu().numpy())
    }

def get_random_samples(folder_path: str, num_samples: int = 5) -> List[Dict]:
    """Get random samples from a folder"""
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    if len(files) < num_samples:
        logger.warning(f"Only {len(files)} files available in {folder_path}, requested {num_samples}")
        num_samples = len(files)
    
    selected_files = random.sample(files, num_samples)
    
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

def find_optimal_threshold(scores: List[float], true_labels: List[int], threshold_range: Tuple[float, float] = (0.01, 0.99)) -> Tuple[float, float]:
    """Find the optimal threshold that maximizes F1 score"""
    start_thresh, end_thresh = threshold_range
    thresholds = np.arange(start_thresh, end_thresh, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
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
    
    return best_threshold, best_f1

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

def cross_validate_model(model, tokenizer, device, samples: List[Dict], model_type: str, threshold: float = None, n_folds: int = 5) -> Dict:
    """Perform cross-validation on the model"""
    # Prepare data
    X = samples
    y = [RELEVANT_LABEL if sample['folder'] == 'relevant' else IRRELEVANT_LABEL for sample in samples]
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_predictions = []
    all_true_labels = []
    all_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Processing fold {fold + 1}/{n_folds}")
        
        test_samples = [X[i] for i in test_idx]
        test_labels = [y[i] for i in test_idx]
        
        fold_predictions = []
        fold_scores = []
        
        for sample in test_samples:
            if model_type == 'classifier':
                result = predict_classifier(sample['content'], model, tokenizer, device)
                fold_predictions.append(result['prediction'])
                fold_scores.append(result['confidence'])
            else:  # regressor
                result = predict_regressor(sample['content'], model, tokenizer, device, threshold or 0.5)
                fold_predictions.append(1 if result['is_relevant'] else 0)
                fold_scores.append(result['score'])
        
        # Calculate fold metrics
        fold_metrics = calculate_metrics(fold_predictions, test_labels)
        fold_results.append(fold_metrics)
        
        # Accumulate for overall metrics
        all_predictions.extend(fold_predictions)
        all_true_labels.extend(test_labels)
        all_scores.extend(fold_scores)
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(all_predictions, all_true_labels)
    
    # Calculate mean and std across folds
    fold_f1s = [result['f1'] for result in fold_results]
    fold_accuracies = [result['accuracy'] for result in fold_results]
    
    # Add regression metrics for regressor models
    if model_type != 'classifier':
        mse = mean_squared_error(all_true_labels, all_scores)
        r2 = r2_score(all_true_labels, all_scores)
        overall_metrics['mse'] = mse
        overall_metrics['r2'] = r2
    
    return {
        'overall_metrics': overall_metrics,
        'fold_results': fold_results,
        'mean_f1': np.mean(fold_f1s),
        'std_f1': np.std(fold_f1s),
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies),
        'all_predictions': all_predictions,
        'all_scores': all_scores,
        'all_true_labels': all_true_labels,
        'threshold': threshold
    }

def main():
    logger.info("Starting robust model comparison with cross-validation...")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"robust_model_comparison_{timestamp}.txt")
    
    # Get test samples
    logger.info(f"Loading {NUM_SAMPLES} samples from each category...")
    irrelevant_samples = get_random_samples(IRRELEVANT_FOLDER, NUM_SAMPLES)
    relevant_samples = get_random_samples(RELEVANT_FOLDER, NUM_SAMPLES)
    
    # Add folder information to samples
    for sample in irrelevant_samples:
        sample['folder'] = 'irrelevant'
    for sample in relevant_samples:
        sample['folder'] = 'relevant'
    
    all_samples = irrelevant_samples + relevant_samples
    logger.info(f"Total samples: {len(all_samples)} ({len(irrelevant_samples)} irrelevant, {len(relevant_samples)} relevant)")
    
    # Load models
    logger.info("Loading models...")
    classifier_model, classifier_tokenizer, device = load_classifier_model(CLASSIFIER_MODEL_PATH)
    regressor_model, regressor_tokenizer, _ = load_regressor_model(REGRESSOR_MODEL_PATH)
    url_regressor_model, url_regressor_tokenizer, _ = load_regressor_model(URL_REGRESSOR_MODEL_PATH)
    
    # Debug: Check score distributions
    logger.info("Analyzing score distributions...")
    debug_scores = {'regressor': [], 'url_regressor': []}
    debug_labels = []
    
    for sample in all_samples[:20]:  # Sample 20 for debugging
        true_label = RELEVANT_LABEL if sample['folder'] == 'relevant' else IRRELEVANT_LABEL
        debug_labels.append(true_label)
        
        reg_result = predict_regressor(sample['content'], regressor_model, regressor_tokenizer, device, 0.5)
        url_reg_result = predict_regressor(sample['content'], url_regressor_model, url_regressor_tokenizer, device, 0.5)
        
        debug_scores['regressor'].append(reg_result['score'])
        debug_scores['url_regressor'].append(url_reg_result['score'])
        
        logger.debug(f"Sample {sample['filename'][:20]}: true={true_label}, reg_score={reg_result['score']:.4f}, url_reg_score={url_reg_result['score']:.4f}")
    
    logger.info(f"Debug score ranges - Regressor: {min(debug_scores['regressor']):.4f} to {max(debug_scores['regressor']):.4f}")
    logger.info(f"Debug score ranges - URL Regressor: {min(debug_scores['url_regressor']):.4f} to {max(debug_scores['url_regressor']):.4f}")
    
    # Find optimal thresholds using a subset
    threshold_samples = all_samples[:100]  # Use subset for threshold finding
    
    logger.info("Finding optimal thresholds...")
    reg_scores = []
    url_reg_scores = []
    threshold_labels = []
    
    for sample in threshold_samples:
        true_label = RELEVANT_LABEL if sample['folder'] == 'relevant' else IRRELEVANT_LABEL
        threshold_labels.append(true_label)
        
        reg_result = predict_regressor(sample['content'], regressor_model, regressor_tokenizer, device, 0.5)
        url_reg_result = predict_regressor(sample['content'], url_regressor_model, url_regressor_tokenizer, device, 0.5)
        
        reg_scores.append(reg_result['score'])
        url_reg_scores.append(url_reg_result['score'])
    
    optimal_reg_threshold, optimal_reg_f1 = find_optimal_threshold(reg_scores, threshold_labels)
    optimal_url_threshold, optimal_url_f1 = find_optimal_threshold(url_reg_scores, threshold_labels)
    
    logger.info(f"Optimal regressor threshold: {optimal_reg_threshold:.4f} (F1: {optimal_reg_f1:.4f})")
    logger.info(f"Optimal URL regressor threshold: {optimal_url_threshold:.4f} (F1: {optimal_url_f1:.4f})")
    logger.info(f"API regressor threshold: {API_REGRESSOR_THRESHOLD} (for comparison)")
    logger.info(f"API URL regressor threshold: {API_URL_REGRESSOR_THRESHOLD} (for comparison)")
    
    # Perform cross-validation
    logger.info("Performing cross-validation...")
    classifier_cv_results = cross_validate_model(classifier_model, classifier_tokenizer, device, all_samples, 'classifier', n_folds=N_FOLDS)
    regressor_cv_results = cross_validate_model(regressor_model, regressor_tokenizer, device, all_samples, 'regressor', optimal_reg_threshold, n_folds=N_FOLDS)
    url_regressor_cv_results = cross_validate_model(url_regressor_model, url_regressor_tokenizer, device, all_samples, 'url_regressor', optimal_url_threshold, n_folds=N_FOLDS)
    
    # Also test with API thresholds
    regressor_api_cv_results = cross_validate_model(regressor_model, regressor_tokenizer, device, all_samples, 'regressor', API_REGRESSOR_THRESHOLD, n_folds=N_FOLDS)
    url_regressor_api_cv_results = cross_validate_model(url_regressor_model, url_regressor_tokenizer, device, all_samples, 'url_regressor', API_URL_REGRESSOR_THRESHOLD, n_folds=N_FOLDS)
    
    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"ROBUST MODEL COMPARISON RESULTS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Configuration:\n")
        f.write(f"- Total samples: {len(all_samples)} ({len(irrelevant_samples)} irrelevant, {len(relevant_samples)} relevant)\n")
        f.write(f"- Cross-validation folds: {N_FOLDS}\n")
        f.write(f"- Evaluation method: Stratified K-Fold Cross-Validation\n\n")
        
        f.write("Threshold Analysis:\n")
        f.write(f"- Optimal regressor threshold: {optimal_reg_threshold:.4f} (F1: {optimal_reg_f1:.4f})\n")
        f.write(f"- Optimal URL regressor threshold: {optimal_url_threshold:.4f} (F1: {optimal_url_f1:.4f})\n")
        f.write(f"- API regressor threshold: {API_REGRESSOR_THRESHOLD}\n")
        f.write(f"- API URL regressor threshold: {API_URL_REGRESSOR_THRESHOLD}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Classifier results
        f.write("1. CLASSIFIER MODEL\n")
        f.write("-" * 30 + "\n")
        metrics = classifier_cv_results['overall_metrics']
        f.write(f"Mean Accuracy: {classifier_cv_results['mean_accuracy']:.4f} ± {classifier_cv_results['std_accuracy']:.4f}\n")
        f.write(f"Mean F1 Score: {classifier_cv_results['mean_f1']:.4f} ± {classifier_cv_results['std_f1']:.4f}\n")
        f.write(f"Overall Precision: {metrics['precision']:.4f}\n")
        f.write(f"Overall Recall: {metrics['recall']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}\n\n")
        
        # Regressor results (optimal threshold)
        f.write("2. REGRESSOR MODEL (Optimal Threshold)\n")
        f.write("-" * 40 + "\n")
        metrics = regressor_cv_results['overall_metrics']
        f.write(f"Threshold: {regressor_cv_results['threshold']:.4f}\n")
        f.write(f"Mean Accuracy: {regressor_cv_results['mean_accuracy']:.4f} ± {regressor_cv_results['std_accuracy']:.4f}\n")
        f.write(f"Mean F1 Score: {regressor_cv_results['mean_f1']:.4f} ± {regressor_cv_results['std_f1']:.4f}\n")
        f.write(f"Overall Precision: {metrics['precision']:.4f}\n")
        f.write(f"Overall Recall: {metrics['recall']:.4f}\n")
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}\n\n")
        
        # Regressor results (API threshold)
        f.write("3. REGRESSOR MODEL (API Threshold)\n")
        f.write("-" * 35 + "\n")
        metrics = regressor_api_cv_results['overall_metrics']
        f.write(f"Threshold: {regressor_api_cv_results['threshold']:.4f}\n")
        f.write(f"Mean Accuracy: {regressor_api_cv_results['mean_accuracy']:.4f} ± {regressor_api_cv_results['std_accuracy']:.4f}\n")
        f.write(f"Mean F1 Score: {regressor_api_cv_results['mean_f1']:.4f} ± {regressor_api_cv_results['std_f1']:.4f}\n")
        f.write(f"Overall Precision: {metrics['precision']:.4f}\n")
        f.write(f"Overall Recall: {metrics['recall']:.4f}\n")
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}\n\n")
        
        # URL Regressor results (optimal threshold)
        f.write("4. URL REGRESSOR MODEL (Optimal Threshold)\n")
        f.write("-" * 44 + "\n")
        metrics = url_regressor_cv_results['overall_metrics']
        f.write(f"Threshold: {url_regressor_cv_results['threshold']:.4f}\n")
        f.write(f"Mean Accuracy: {url_regressor_cv_results['mean_accuracy']:.4f} ± {url_regressor_cv_results['std_accuracy']:.4f}\n")
        f.write(f"Mean F1 Score: {url_regressor_cv_results['mean_f1']:.4f} ± {url_regressor_cv_results['std_f1']:.4f}\n")
        f.write(f"Overall Precision: {metrics['precision']:.4f}\n")
        f.write(f"Overall Recall: {metrics['recall']:.4f}\n")
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}\n\n")
        
        # URL Regressor results (API threshold)
        f.write("5. URL REGRESSOR MODEL (API Threshold)\n")
        f.write("-" * 39 + "\n")
        metrics = url_regressor_api_cv_results['overall_metrics']
        f.write(f"Threshold: {url_regressor_api_cv_results['threshold']:.4f}\n")
        f.write(f"Mean Accuracy: {url_regressor_api_cv_results['mean_accuracy']:.4f} ± {url_regressor_api_cv_results['std_accuracy']:.4f}\n")
        f.write(f"Mean F1 Score: {url_regressor_api_cv_results['mean_f1']:.4f} ± {url_regressor_api_cv_results['std_f1']:.4f}\n")
        f.write(f"Overall Precision: {metrics['precision']:.4f}\n")
        f.write(f"Overall Recall: {metrics['recall']:.4f}\n")
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}\n\n")
        
        # Summary comparison
        f.write("="*80 + "\n")
        f.write("SUMMARY COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        results = [
            ("Classifier", classifier_cv_results),
            ("Regressor (Optimal)", regressor_cv_results),
            ("Regressor (API)", regressor_api_cv_results),
            ("URL Regressor (Optimal)", url_regressor_cv_results),
            ("URL Regressor (API)", url_regressor_api_cv_results)
        ]
        
        f.write(f"{'Model':<25} {'Mean F1':<15} {'Std F1':<15} {'Mean Accuracy':<15}\n")
        f.write("-" * 75 + "\n")
        for name, result in results:
            f.write(f"{name:<25} {result['mean_f1']:.4f} ± {result['std_f1']:.4f}  {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}\n")
    
    # Print summary to console
    print("\n" + "="*80)
    print("ROBUST MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Mean F1':<15} {'Mean Accuracy':<15}")
    print("-" * 60)
    for name, result in results:
        print(f"{name:<25} {result['mean_f1']:.4f} ± {result['std_f1']:.4f}  {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    
    # Find best model
    best_f1 = max(result['mean_f1'] for _, result in results)
    best_model = next(name for name, result in results if result['mean_f1'] == best_f1)
    print(f"\nBest Model: {best_model} (Mean F1: {best_f1:.4f})")
    
    print(f"\nDetailed results saved to: {output_file}")
    logger.info(f"Robust model comparison completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()