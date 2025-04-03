import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import glob

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for training")

# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Function to load posts from folders
def load_posts_from_folders(relevant_folder, irrelevant_folder):
    posts = []
    labels = []
    
    # Load relevant posts
    for filepath in glob.glob(os.path.join(relevant_folder, "*.txt")):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:  # Make sure we don't add empty posts
                    posts.append(content)
                    labels.append(1)  # 1 for relevant
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    # Load irrelevant posts
    for filepath in glob.glob(os.path.join(irrelevant_folder, "*.txt")):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:  # Make sure we don't add empty posts
                    posts.append(content)
                    labels.append(0)  # 0 for irrelevant
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    print(f"Loaded {len(posts)} posts in total:")
    print(f"- {labels.count(1)} relevant posts")
    print(f"- {labels.count(0)} irrelevant posts")
    
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
def train_model(model, train_loader, val_loader, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
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
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(val_true, val_preds)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(classification_report(val_true, val_preds))
    
    return model

# Function to predict relevance of new posts
def predict_relevance(new_posts, model, tokenizer, device):
    model.eval()
    predictions = []
    
    for post in new_posts:
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
    
    return predictions

# Function to predict relevance for new posts from a folder
def predict_folder(folder_path, model, tokenizer, device):
    new_posts = []
    filenames = []
    
    for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    new_posts.append(content)
                    filenames.append(os.path.basename(filepath))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
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
    
    print(f"Training set: {len(train_texts)} posts")
    print(f"Validation set: {len(val_texts)} posts")
    
    # Create datasets
    train_dataset = RedditDataset(train_texts, train_labels, tokenizer)
    val_dataset = RedditDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train model
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, epochs=3)
    
    # Save model
    model_save_path = "reddit_topic_classifier.pt"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Optional: Test on new data
    test_folder = "test_posts"  # Folder with posts you want to classify
    if os.path.exists(test_folder):
        print(f"\nClassifying posts in {test_folder}...")
        test_predictions = predict_folder(test_folder, trained_model, tokenizer, device)
        
        # Display results
        print("\nResults:")
        for pred in test_predictions:
            print(f"File: {pred['filename']}")
            print(f"Relevant: {pred['relevant']}")
            print(f"Confidence: {pred['confidence']:.4f}")
            print()

if __name__ == "__main__":
    main()
