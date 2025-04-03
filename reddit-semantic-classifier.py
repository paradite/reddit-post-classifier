import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import glob
import wandb
import time
import logging

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
model_name = 'distilbert-base-uncased'
logger.info(f"Loading model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
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
                    labels.append(1)  # 1 for relevant
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
                    labels.append(0)  # 0 for irrelevant
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
    
    logger.info(f"Loaded {len(posts)} posts in total:")
    logger.info(f"- {labels.count(1)} relevant posts")
    logger.info(f"- {labels.count(0)} irrelevant posts")
    
    return posts, labels

# Custom dataset class
class RedditDataset(Dataset):
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training function
def train_model(model, train_loader, val_loader, epochs=3, project_name="reddit-classifier"):
    # Initialize wandb
    wandb.init(project=project_name, config={
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": 2e-5,
        "device": str(device),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset)
    })
    
    # Log model architecture
    wandb.watch(model, log="all")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose="True")
    
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        logger.info(f"Training on {len(train_loader)} batches")
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Calculate accuracy
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            loss.backward()
            optimizer.step()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                batch_time = time.time() - batch_start_time
                current_accuracy = 100. * train_correct / train_total
                logger.info(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}, Accuracy: {current_accuracy:.2f}%, Time: {batch_time:.2f}s")
                
                # Log batch metrics to wandb
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_accuracy": current_accuracy,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        logger.info(f"Epoch {epoch + 1} Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0
        
        logger.info("Starting validation")
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            val_loss += outputs.loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())
            
            # Print validation progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Validation batch {batch_idx + 1}/{len(val_loader)} - Loss: {outputs.loss.item():.4f}")
        
        accuracy = accuracy_score(val_true, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(val_true, val_preds))
        
        # Update learning rate scheduler
        scheduler.step(accuracy)
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": accuracy,
            "epoch_time": time.time() - epoch_start_time
        })
        
        # Save best model
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            logger.info(f"New best model with validation accuracy: {best_val_accuracy:.4f}")
            torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}.pt")
            wandb.save(f"best_model_epoch_{epoch+1}.pt")
    
    wandb.finish()
    return model

# Function to predict relevance of new posts
def predict_relevance(new_posts, model, tokenizer, device):
    model.eval()
    predictions = []
    
    logger.info(f"Predicting relevance for {len(new_posts)} posts")
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
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
            confidence = probabilities[0][prediction].cpu().numpy()
        
        predictions.append({
            'post': post[:100] + "..." if len(post) > 100 else post,  # Truncate for display
            'relevant': bool(prediction),
            'confidence': float(confidence)
        })
        
        if (i+1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(new_posts)} posts")
    
    return predictions

# Function to predict relevance for new posts from a folder
def predict_folder(folder_path, model, tokenizer, device):
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
    predictions = predict_relevance(new_posts, model, tokenizer, device)
    
    # Add filenames to predictions
    for i, pred in enumerate(predictions):
        pred['filename'] = filenames[i]
    
    return predictions

# Main execution function
def main():
    # Paths to your folders
    relevant_folder = "relevant_posts"
    irrelevant_folder = "irrelevant_posts"
    
    # Load data
    posts, labels = load_posts_from_folders(relevant_folder, irrelevant_folder)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        posts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training set: {len(train_texts)} posts")
    logger.info(f"Validation set: {len(val_texts)} posts")
    
    # Create datasets
    train_dataset = RedditDataset(train_texts, train_labels, tokenizer)
    val_dataset = RedditDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train model
    logger.info("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, epochs=3)
    
    # Save model
    model_save_path = "reddit_topic_classifier.pt"
    torch.save(trained_model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # Optional: Test on new data
    test_folder = "test_posts"  # Folder with posts you want to classify
    if os.path.exists(test_folder):
        logger.info(f"\nClassifying posts in {test_folder}...")
        test_predictions = predict_folder(test_folder, trained_model, tokenizer, device)
        
        # Display results
        logger.info("\nResults:")
        for pred in test_predictions:
            logger.info(f"File: {pred['filename']}")
            logger.info(f"Relevant: {pred['relevant']}")
            logger.info(f"Confidence: {pred['confidence']:.4f}")
            logger.info("---")

if __name__ == "__main__":
    main()
