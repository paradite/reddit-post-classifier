# In a new file, e.g., simple_api_server.py

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import time
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
# MODEL_PATH = "reddit_topic_classifier_run3.pt"
CLASSIFIER_MODEL_PATH = "best_model_run12_epoch_9.pt"
REGRESSOR_MODEL_PATH = "best_regressor_run1_epoch_4.pt"
URL_REGRESSOR_MODEL_PATH = "best_url_regressor_run1_epoch_5.pt"
MODEL_NAME = "roberta-base"  # Changed from distilbert-base-uncased to roberta-base
MAX_BATCH_SIZE = 10
THRESHOLD = 0.05  # Threshold for regressor model
URL_THRESHOLD = 0.15  # Threshold for URL regressor model

# Load models
logging.info("Loading models and tokenizer...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define the RegressionHead class
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

# Load classifier model
logging.info("Loading classifier model...")
classifier_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
classifier_model.to(device)

# Load state dict with compatibility handling for classifier
classifier_state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=device)

# Filter out incompatible keys for classifier
filtered_classifier_state_dict = {}
for key, value in classifier_state_dict.items():
    # Skip keys that are specific to DistilBERT but not in RoBERTa
    if "roberta.embeddings.position_ids" in key:
        continue
    
    # Map DistilBERT keys to RoBERTa keys if needed
    if "distilbert" in key:
        new_key = key.replace("distilbert", "roberta")
        filtered_classifier_state_dict[new_key] = value
    else:
        filtered_classifier_state_dict[key] = value

# Load the filtered state dict for classifier
classifier_model.load_state_dict(filtered_classifier_state_dict, strict=False)
classifier_model.eval()

# Load regressor model
logging.info("Loading regressor model...")
regressor_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
regressor_model.classifier = RegressionHead(regressor_model.config)
regressor_model.to(device)

# Load URL regressor model
logging.info("Loading URL regressor model...")
url_regressor_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
url_regressor_model.classifier = RegressionHead(url_regressor_model.config)
url_regressor_model.to(device)

# Load state dict with compatibility handling for regressor
regressor_state_dict = torch.load(REGRESSOR_MODEL_PATH, map_location=device)
url_regressor_state_dict = torch.load(URL_REGRESSOR_MODEL_PATH, map_location=device)

# Filter out incompatible keys for regressor
filtered_regressor_state_dict = {}
filtered_url_regressor_state_dict = {}

for key, value in regressor_state_dict.items():
    # Skip keys that are specific to DistilBERT but not in RoBERTa
    if "roberta.embeddings.position_ids" in key:
        continue
    
    # Map DistilBERT keys to RoBERTa keys if needed
    if "distilbert" in key:
        new_key = key.replace("distilbert", "roberta")
        filtered_regressor_state_dict[new_key] = value
    else:
        filtered_regressor_state_dict[key] = value

for key, value in url_regressor_state_dict.items():
    # Skip keys that are specific to DistilBERT but not in RoBERTa
    if "roberta.embeddings.position_ids" in key:
        continue
    
    # Map DistilBERT keys to RoBERTa keys if needed
    if "distilbert" in key:
        new_key = key.replace("distilbert", "roberta")
        filtered_url_regressor_state_dict[new_key] = value
    else:
        filtered_url_regressor_state_dict[key] = value

# Load the filtered state dict for regressor
regressor_model.load_state_dict(filtered_regressor_state_dict, strict=False)
regressor_model.eval()

# Load the filtered state dict for URL regressor
url_regressor_model.load_state_dict(filtered_url_regressor_state_dict, strict=False)
url_regressor_model.eval()

logging.info("Models loaded successfully")

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        start_time = time.time()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            # Handle both single content and bulk content
            if 'content' in data:
                # Single content request (backward compatibility)
                items = [{
                    'content': data.get('content', ''),
                    'url': data.get('url', '')
                }]
            elif 'contents' in data:
                # Bulk content request
                if isinstance(data['contents'], list):
                    # New format: array of dicts
                    if all(isinstance(item, dict) for item in data['contents']):
                        items = data['contents']
                    # Old format: array of strings
                    else:
                        items = [{
                            'content': content,
                            'url': url
                        } for content, url in zip(
                            data['contents'],
                            data.get('urls', [''] * len(data['contents']))
                        )]
                else:
                    raise ValueError("'contents' must be a list")
                
                if len(items) > MAX_BATCH_SIZE:
                    error_msg = f"Request exceeded maximum batch size of {MAX_BATCH_SIZE}. Received {len(items)} items."
                    logging.error(error_msg)
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": error_msg}).encode('utf-8'))
                    return
            else:
                raise ValueError("Request must contain either 'content' or 'contents' field")
                
            logging.info(f"Received request with {len(items)} item(s)")
        except Exception as e:
            logging.error(f"Failed to parse request data: {str(e)}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return

        results = []
        for i, item in enumerate(items):
            text_start_time = time.time()
            text = item['content']
            url = item.get('url', '')
            
            # Tokenize text for classifier and regressor models
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = torch.tensor(encoding['input_ids']).to(device)
            attention_mask = torch.tensor(encoding['attention_mask']).to(device)

            # Get classifier prediction
            with torch.no_grad():
                classifier_output = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
                classifier_logits = classifier_output.logits
                classifier_probs = torch.softmax(classifier_logits, dim=1)
                classifier_pred = int(torch.argmax(classifier_probs, dim=1).item())
                classifier_conf = classifier_probs[0][classifier_pred].item()

            # Get regressor prediction
            with torch.no_grad():
                regressor_output = regressor_model(input_ids=input_ids, attention_mask=attention_mask)
                regressor_logits = regressor_output.logits.squeeze()
                # Apply sigmoid to get 0-1 range
                regressor_score = torch.sigmoid(regressor_logits).cpu().numpy()
                # Handle both scalar and array outputs
                if isinstance(regressor_score, np.ndarray) and regressor_score.size > 1:
                    regressor_score = regressor_score[0]
                # Determine binary classification based on threshold
                regressor_is_relevant = regressor_score >= THRESHOLD

            # Get URL regressor prediction if URL is provided
            url_regressor_result = None
            if url:
                # Prepare text with URL prefix for URL regressor
                processed_text = f"{url}\n\n{text}"
                url_encoding = tokenizer(
                    processed_text,
                    add_special_tokens=True,
                    max_length=512,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                url_input_ids = torch.tensor(url_encoding['input_ids']).to(device)
                url_attention_mask = torch.tensor(url_encoding['attention_mask']).to(device)
                
                with torch.no_grad():
                    url_regressor_output = url_regressor_model(input_ids=url_input_ids, attention_mask=url_attention_mask)
                    url_regressor_logits = url_regressor_output.logits.squeeze()
                    # Apply sigmoid to get 0-1 range
                    url_regressor_score = torch.sigmoid(url_regressor_logits).cpu().numpy()
                    # Handle both scalar and array outputs
                    if isinstance(url_regressor_score, np.ndarray) and url_regressor_score.size > 1:
                        url_regressor_score = url_regressor_score[0]
                    # Determine binary classification based on threshold
                    url_regressor_is_relevant = url_regressor_score >= URL_THRESHOLD
                    url_regressor_result = {
                        'score': float(url_regressor_score),
                        'is_relevant': bool(url_regressor_is_relevant),
                        'threshold': URL_THRESHOLD
                    }

            # Combine results
            result = {
                'classifier': {
                    'relevant': bool(classifier_pred == 1), 
                    'confidence': float(classifier_conf)
                },
                'regressor': {
                    'score': float(regressor_score),
                    'is_relevant': bool(regressor_is_relevant),
                    'threshold': THRESHOLD
                }
            }
            
            # Add URL regressor results if available
            if url_regressor_result:
                result['url_regressor'] = url_regressor_result
            
            # For backward compatibility, include the original fields
            result['relevant'] = bool(classifier_pred == 1)
            result['confidence'] = float(classifier_conf)
            
            results.append(result)
            text_time = time.time() - text_start_time
            logging.info(f"Item {i+1}/{len(items)}: C:{'Y' if classifier_pred == 1 else 'N'}({classifier_conf:.2f}) R:{'Y' if regressor_is_relevant else 'N'}({regressor_score:.4f})" + (f" U:{'Y' if url_regressor_result['is_relevant'] else 'N'}({url_regressor_result['score']:.4f})" if url_regressor_result else "") + f" t:{text_time:.3f}s")

        # For backward compatibility, if it was a single content request, return the same format
        if 'content' in data:
            response = results[0]
        else:
            response = {'results': results}

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))
        
        total_time = time.time() - start_time
        logging.info(f"Request completed in {total_time:.3f}s")

def run():
    server_address = ('0.0.0.0', 8080)
    httpd = HTTPServer(server_address, Handler)
    logging.info(f"Server started on {server_address[0]}:{server_address[1]}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
