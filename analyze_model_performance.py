import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics."""
    metrics = {}
    
    # Convert to pandas Series if not already
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    
    # Remove entries where either true or pred is null
    mask = ~(y_true.isna() | y_pred.isna())
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'accuracy': 0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'true_negatives': 0
        }
    
    # Calculate metrics
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Extract values from confusion matrix
    metrics['true_negatives'] = cm[0, 0]
    metrics['false_positives'] = cm[0, 1]
    metrics['false_negatives'] = cm[1, 0]
    metrics['true_positives'] = cm[1, 1]
    
    return metrics

def main():
    # Read the CSV file
    df = pd.read_csv('Supabase Snippet Retrieve Tracking Hits for Team 1 result.csv')
    
    # Convert status to binary (RELEVANT/REPLIED = 1, others = 0)
    df['true_label'] = df['status'].isin(['RELEVANT', 'REPLIED']).astype(int)
    
    # Convert boolean columns to int, handling NaN values
    df['ai_relevant'] = df['ai_relevant'].fillna(False).astype(int)
    df['ai_regressor_relevant'] = df['ai_regressor_relevant'].fillna(False).astype(int)
    
    # Calculate metrics for classifier model
    classifier_metrics = calculate_metrics(df['true_label'], df['ai_relevant'])
    
    # Calculate metrics for regressor model
    regressor_metrics = calculate_metrics(df['true_label'], df['ai_regressor_relevant'])
    
    # Print results
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print('----------------------------------------')
    print("\nClassifier Model (ai_relevant) Performance:")
    print("----------------------------------------")
    print(f"Precision: {classifier_metrics['precision']:.3f}")
    print(f"Recall: {classifier_metrics['recall']:.3f}")
    print(f"F1 Score: {classifier_metrics['f1']:.3f}")
    print(f"Accuracy: {classifier_metrics['accuracy']:.3f}")
    print("\nError Analysis:")
    print(f"False Positives (incorrectly marked as relevant): {classifier_metrics['false_positives']}")
    print(f"False Negatives (missed relevant posts): {classifier_metrics['false_negatives']}")
    
    print("\nRegressor Model (ai_regressor_relevant) Performance:")
    print("----------------------------------------")
    print(f"Precision: {regressor_metrics['precision']:.3f}")
    print(f"Recall: {regressor_metrics['recall']:.3f}")
    print(f"F1 Score: {regressor_metrics['f1']:.3f}")
    print(f"Accuracy: {regressor_metrics['accuracy']:.3f}")
    print("\nError Analysis:")
    print(f"False Positives (incorrectly marked as relevant): {regressor_metrics['false_positives']}")
    print(f"False Negatives (missed relevant posts): {regressor_metrics['false_negatives']}")
    
    # Calculate and print additional statistics
    print("\nAdditional Statistics:")
    print("----------------------------------------")
    print(f"Total samples: {len(df)}")
    print(f"Number of relevant posts (RELEVANT/REPLIED): {df['true_label'].sum()}")
    print(f"Number of irrelevant posts: {len(df) - df['true_label'].sum()}")
    print(f"Number of null predictions in classifier: {df['ai_relevant'].isna().sum()}")
    print(f"Number of null predictions in regressor: {df['ai_regressor_relevant'].isna().sum()}")

if __name__ == "__main__":
    main() 