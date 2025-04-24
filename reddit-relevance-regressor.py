import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import glob
import wandb
import time
import logging

# Global run number for tracking different training runs
RUN_NUMBER = 1  # Increment this for each new training run

# Define label constants
IRRELEVANT_LABEL = 0.0  # Posts that are not relevant to the topic
RELEVANT_LABEL = 1.0    # Posts that are relevant to the topic

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} for training")

# Load pre-trained model and tokenizer
model_name = 'roberta-base'
logger.info(f"Loading model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# For regression, we'll use a model with a single output
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
# Replace the classification head with a regression head
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

model.classifier = RegressionHead(model.config)
model.to(device)
logger.info(f"Model loaded and moved to {device}")

# Function to load posts from folders
def load_posts_from_folders(relevant_folder, irrelevant_folder):
    posts = []
    labels = []
    
    # Load relevant posts
    logger.info(f"Loading relevant posts from {relevant_folder}")
    for filepath in glob.glob(os.path.join(relevant_folder, "*.txt")):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:  # Make sure we don't add empty posts
                    posts.append(content)
                    labels.append(RELEVANT_LABEL)  # 1.0 for relevant
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
    
    # Load irrelevant posts
    logger.info(f"Loading irrelevant posts from {irrelevant_folder}")
    for filepath in glob.glob(os.path.join(irrelevant_folder, "*.txt")):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:  # Make sure we don't add empty posts
                    posts.append(content)
                    labels.append(IRRELEVANT_LABEL)  # 0.0 for irrelevant
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
    
    logger.info(f"Loaded {len(posts)} posts in total:")
    logger.info(f"- {sum(1 for l in labels if l == RELEVANT_LABEL)} relevant posts")
    logger.info(f"- {sum(1 for l in labels if l == IRRELEVANT_LABEL)} irrelevant posts")
    
    return posts, labels

# Custom dataset class for regression
class RedditRegressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Training function with early stopping
def train_model(model, train_loader, val_loader, epochs=10, patience=3, project_name="reddit-regressor"):
    # Initialize wandb
    wandb.init(project=project_name, config={
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": 2e-5,
        "device": str(device),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "patience": patience,
        "run_number": RUN_NUMBER
    })
    
    # Log model architecture
    wandb.watch(model, log="all")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    best_val_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_predictions = []
        train_true = []
        
        logger.info(f"Training on {len(train_loader)} batches")
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # For regression, we use MSE loss
            logits = outputs.logits.squeeze()
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits, labels)
            
            train_loss += loss.item()
            
            # Store predictions and true values for metrics
            train_predictions.extend(logits.detach().cpu().numpy())
            train_true.extend(labels.cpu().numpy())
            
            loss.backward()
            optimizer.step()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                batch_time = time.time() - batch_start_time
                current_mse = loss.item()
                logger.info(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {current_mse:.4f}, Time: {batch_time:.2f}s")
                
                # Log batch metrics to wandb
                wandb.log({
                    "batch_loss": current_mse,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        avg_train_loss = train_loss / len(train_loader)
        train_mse = mean_squared_error(train_true, train_predictions)
        train_r2 = r2_score(train_true, train_predictions)
        logger.info(f"Epoch {epoch + 1} Training - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        
        # Validation
        model.eval()
        val_predictions = []
        val_true = []
        val_loss = 0
        
        logger.info("Starting validation")
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits.squeeze()
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits, labels)
                
                val_loss += loss.item()
                val_predictions.extend(logits.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
            
            # Print validation progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Validation batch {batch_idx + 1}/{len(val_loader)} - Loss: {loss.item():.4f}")
        
        val_mse = mean_squared_error(val_true, val_predictions)
        val_r2 = r2_score(val_true, val_predictions)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Validation - MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_mse)
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_mse": train_mse,
            "train_r2": train_r2,
            "val_loss": avg_val_loss,
            "val_mse": val_mse,
            "val_r2": val_r2,
            "epoch_time": time.time() - epoch_start_time
        })
        
        # Early stopping check
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0
            logger.info(f"New best model with validation MSE: {best_val_mse:.4f}")
            torch.save(model.state_dict(), f"best_regressor_run{RUN_NUMBER}_epoch_{epoch+1}.pt")
            wandb.save(f"best_regressor_run{RUN_NUMBER}_epoch_{epoch+1}.pt")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs. Best MSE: {best_val_mse:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs. Best epoch was {best_epoch+1} with MSE {best_val_mse:.4f}")
                break
    
    wandb.finish()
    return model

# Function to predict relevance score of new posts
def predict_relevance_score(new_posts, model, tokenizer, device, threshold=0.5):
    model.eval()
    predictions = []
    
    logger.info(f"Predicting relevance scores for {len(new_posts)} posts")
    for i, post in enumerate(new_posts):
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
            score = torch.sigmoid(logits).cpu().numpy()[0]  # Apply sigmoid to get 0-1 range
            # Determine binary classification based on threshold
            is_relevant = score >= threshold
        
        predictions.append({
            'post': post[:100] + "..." if len(post) > 100 else post,  # Truncate for display
            'relevance_score': float(score),
            'relevant': bool(is_relevant),
            'threshold': threshold
        })
        
        if (i+1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(new_posts)} posts")
    
    return predictions

# Function to predict relevance scores for new posts from a folder
def predict_folder(folder_path, model, tokenizer, device, threshold=0.5):
    new_posts = []
    filenames = []
    
    logger.info(f"Loading posts from {folder_path}")
    for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    new_posts.append(content)
                    filenames.append(os.path.basename(filepath))
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
    
    logger.info(f"Loaded {len(new_posts)} posts from {folder_path}")
    predictions = predict_relevance_score(new_posts, model, tokenizer, device, threshold)
    
    # Add filenames to predictions
    for i, pred in enumerate(predictions):
        pred['filename'] = filenames[i]
    
    return predictions

# Function to find optimal threshold
def find_optimal_threshold(val_predictions, val_true):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        binary_preds = [1 if p >= threshold else 0 for p in val_predictions]
        binary_true = [1 if t >= 0.5 else 0 for t in val_true]
        
        # Calculate metrics
        tp = sum(1 for p, t in zip(binary_preds, binary_true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(binary_preds, binary_true) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(binary_preds, binary_true) if p == 0 and t == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
    return best_threshold

# Main execution function
def main():
    # Paths to your folders
    relevant_folder = "relevant_posts"
    irrelevant_folder = "irrelevant_posts"
    
    # Load data
    posts, labels = load_posts_from_folders(relevant_folder, irrelevant_folder)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        posts, labels, test_size=0.2, random_state=42, stratify=[1 if l == RELEVANT_LABEL else 0 for l in labels]
    )
    
    logger.info(f"Training set: {len(train_texts)} posts")
    logger.info(f"Validation set: {len(val_texts)} posts")
    
    # Create datasets
    train_dataset = RedditRegressionDataset(train_texts, train_labels, tokenizer)
    val_dataset = RedditRegressionDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train model with early stopping
    logger.info("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, epochs=10, patience=3)
    
    # Save model
    model_save_path = f"reddit_relevance_regressor_run{RUN_NUMBER}.pt"
    torch.save(trained_model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # Find optimal threshold
    val_predictions = []
    val_true = []
    
    trained_model.eval()
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = trained_model(input_ids=input_ids, attention_mask=attention_mask)
            scores = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
            
            val_predictions.extend(scores.flatten())
            val_true.extend(labels.cpu().numpy())
    
    optimal_threshold = find_optimal_threshold(val_predictions, val_true)
    
    # Optional: Test on new data
    test_folder = "test_posts"  # Folder with posts you want to classify
    if os.path.exists(test_folder):
        logger.info(f"\nClassifying posts in {test_folder}...")
        test_predictions = predict_folder(test_folder, trained_model, tokenizer, device, float(optimal_threshold))
        
        # Display results
        logger.info("\nResults:")
        for pred in test_predictions:
            logger.info(f"File: {pred['filename']}")
            logger.info(f"Relevance Score: {pred['relevance_score']:.4f}")
            logger.info(f"Relevant: {pred['relevant']} (threshold: {pred['threshold']:.2f})")
            logger.info("---")

if __name__ == "__main__":
    main() 