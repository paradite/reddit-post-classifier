import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import time
import numpy as np
from typing import Dict, Optional
import psutil
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants - Top 3 performing models
URL_REGRESSOR_MAY_2025_MODEL_PATH = "reddit-url-regressor-may-2025_run1_epoch_6.pt"
URL_REGRESSOR_MODEL_PATH = "best_url_regressor_run1_epoch_5.pt"
MODEL_NAME = "roberta-base"
MAX_BATCH_SIZE = 10
# Thresholds based on model comparison results
URL_MAY_2025_API_THRESHOLD = 0.15  # Best performing model at API threshold
URL_MAY_2025_OPTIMAL_THRESHOLD = 0.0342  # Best performing model at optimal threshold
URL_REGRESSOR_OPTIMAL_THRESHOLD = 0.1203  # Third best model at optimal threshold

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # RSS in MB
        'vms': memory_info.vms / 1024 / 1024,  # VMS in MB
        'gpu': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0  # GPU memory in MB
    }

# Model Manager class to handle dynamic loading/unloading
class ModelManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Initializing ModelManager with device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.models: Dict[str, Optional[torch.nn.Module]] = {
            'url_regressor_may_2025': None,
            'url_regressor_may_2025_optimal': None,
            'url_regressor_optimal': None
        }
        self.model_paths = {
            'url_regressor_may_2025': URL_REGRESSOR_MAY_2025_MODEL_PATH,
            'url_regressor_may_2025_optimal': URL_REGRESSOR_MAY_2025_MODEL_PATH,
            'url_regressor_optimal': URL_REGRESSOR_MODEL_PATH
        }
        self.active_requests = 0
        logging.info("ModelManager initialized successfully")

    def load_model(self, model_type: str) -> torch.nn.Module:
        if self.models[model_type] is None:
            logging.info(f"Loading {model_type} model...")
            start_time = time.time()
            memory_before = get_memory_usage()
            
            # All top 3 models are regression models
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
            model.classifier = RegressionHead(model.config)

            model.to(self.device)
            
            # Load state dict with compatibility handling
            state_dict = torch.load(self.model_paths[model_type], map_location=self.device)
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
            self.models[model_type] = model
            
            memory_after = get_memory_usage()
            load_time = time.time() - start_time
            logging.info(f"{model_type} model loaded successfully in {load_time:.2f} seconds")
            logging.info(f"Memory usage - Before: {memory_before}, After: {memory_after}")
        else:
            logging.debug(f"{model_type} model already loaded")
        
        return self.models[model_type]

    def unload_model(self, model_type: str):
        if self.models[model_type] is not None:
            logging.info(f"Unloading {model_type} model...")
            memory_before = get_memory_usage()
            
            self.models[model_type] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            memory_after = get_memory_usage()
            logging.info(f"{model_type} model unloaded")
            logging.info(f"Memory usage - Before: {memory_before}, After: {memory_after}")

    def unload_models(self):
        logging.info("Unloading all models...")
        start_time = time.time()
        unloaded_count = 0
        
        for model_type in self.models:
            if self.models[model_type] is not None:
                self.unload_model(model_type)
                unloaded_count += 1
        
        unload_time = time.time() - start_time
        logging.info(f"Unloaded {unloaded_count} models in {unload_time:.2f} seconds")

    def start_request(self):
        self.active_requests += 1
        logging.info(f"Starting request #{self.active_requests}")
        logging.info(f"Memory usage at start: {get_memory_usage()}")

    def end_request(self):
        self.active_requests -= 1
        logging.info(f"Completed request. Active requests remaining: {self.active_requests}")
        logging.info(f"Memory usage at end: {get_memory_usage()}")
        if self.active_requests == 0:
            self.unload_models()

# Initialize model manager
model_manager = ModelManager()

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

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        start_time = time.time()
        model_manager.start_request()
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Handle bulk content
            if 'contents' in data:
                if not isinstance(data['contents'], list):
                    self.send_response(400)
                    self.end_headers()
                    model_manager.end_request()
                    return
                
                if len(data['contents']) > MAX_BATCH_SIZE:
                    self.send_response(400)
                    self.end_headers()
                    model_manager.end_request()
                    return
                
                items = data['contents']
                logging.info(f"Processing bulk request with {len(items)} items")
            else:
                self.send_response(400)
                self.end_headers()
                model_manager.end_request()
                return
                
            logging.info(f"Received request with {len(items)} item(s)")
        except Exception as e:
            logging.error(f"Failed to parse request data: {str(e)}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            model_manager.end_request()
            return

        # Prepare all encodings first
        encodings = []
        url_encodings = []
        for item in items:
            text = item['content']
            url = item.get('url', '')
            
            # Tokenize text
            encoding = model_manager.tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            encodings.append(encoding)
            
            # Prepare URL encoding if URL is provided
            if url:
                processed_text = f"{url}\n\n{text}"
                url_encoding = model_manager.tokenizer(
                    processed_text,
                    add_special_tokens=True,
                    max_length=512,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                url_encodings.append((url, url_encoding))
            else:
                url_encodings.append(None)

        # Process all items with top 3 models
        
        # 1. URL Regressor May 2025 (API threshold) - Best performing model
        may_2025_api_model = model_manager.load_model('url_regressor_may_2025')
        may_2025_api_results = []
        for i, item in enumerate(items):
            # Use URL encoding if available, otherwise use regular encoding
            if url_encodings[i] is not None:
                _, encoding = url_encodings[i]
            else:
                encoding = encodings[i]
                
            input_ids = torch.tensor(encoding['input_ids']).to(model_manager.device)
            attention_mask = torch.tensor(encoding['attention_mask']).to(model_manager.device)
            
            with torch.no_grad():
                output = may_2025_api_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits.squeeze()
                score = torch.sigmoid(logits).cpu().numpy()
                if isinstance(score, np.ndarray) and score.size > 1:
                    score = score[0]
                is_relevant = score >= URL_MAY_2025_API_THRESHOLD
                may_2025_api_results.append((score, is_relevant))
        model_manager.unload_model('url_regressor_may_2025')

        # 2. URL Regressor May 2025 (Optimal threshold) - Second best
        may_2025_optimal_model = model_manager.load_model('url_regressor_may_2025_optimal')
        may_2025_optimal_results = []
        for i, item in enumerate(items):
            # Use URL encoding if available, otherwise use regular encoding
            if url_encodings[i] is not None:
                _, encoding = url_encodings[i]
            else:
                encoding = encodings[i]
                
            input_ids = torch.tensor(encoding['input_ids']).to(model_manager.device)
            attention_mask = torch.tensor(encoding['attention_mask']).to(model_manager.device)
            
            with torch.no_grad():
                output = may_2025_optimal_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits.squeeze()
                score = torch.sigmoid(logits).cpu().numpy()
                if isinstance(score, np.ndarray) and score.size > 1:
                    score = score[0]
                is_relevant = score >= URL_MAY_2025_OPTIMAL_THRESHOLD
                may_2025_optimal_results.append((score, is_relevant))
        model_manager.unload_model('url_regressor_may_2025_optimal')

        # 3. URL Regressor (Optimal threshold) - Third best
        url_regressor_optimal_model = model_manager.load_model('url_regressor_optimal')
        url_regressor_optimal_results = []
        for i, item in enumerate(items):
            # Use URL encoding if available, otherwise use regular encoding
            if url_encodings[i] is not None:
                _, encoding = url_encodings[i]
            else:
                encoding = encodings[i]
                
            input_ids = torch.tensor(encoding['input_ids']).to(model_manager.device)
            attention_mask = torch.tensor(encoding['attention_mask']).to(model_manager.device)
            
            with torch.no_grad():
                output = url_regressor_optimal_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits.squeeze()
                score = torch.sigmoid(logits).cpu().numpy()
                if isinstance(score, np.ndarray) and score.size > 1:
                    score = score[0]
                is_relevant = score >= URL_REGRESSOR_OPTIMAL_THRESHOLD
                url_regressor_optimal_results.append((score, is_relevant))
        model_manager.unload_model('url_regressor_optimal')

        # Combine results from top 3 models
        results = []
        for i in range(len(items)):
            may_2025_api_score, may_2025_api_relevant = may_2025_api_results[i]
            may_2025_optimal_score, may_2025_optimal_relevant = may_2025_optimal_results[i]
            url_optimal_score, url_optimal_relevant = url_regressor_optimal_results[i]
            
            result = {
                'url_regressor_may_2025_api': {
                    'score': float(may_2025_api_score),
                    'is_relevant': bool(may_2025_api_relevant),
                    'threshold': URL_MAY_2025_API_THRESHOLD
                },
                'url_regressor_may_2025_optimal': {
                    'score': float(may_2025_optimal_score),
                    'is_relevant': bool(may_2025_optimal_relevant),
                    'threshold': URL_MAY_2025_OPTIMAL_THRESHOLD
                },
                'url_regressor_optimal': {
                    'score': float(url_optimal_score),
                    'is_relevant': bool(url_optimal_relevant),
                    'threshold': URL_REGRESSOR_OPTIMAL_THRESHOLD
                }
            }
            
            # Use best performing model (May 2025 API) as primary prediction
            result['relevant'] = bool(may_2025_api_relevant)
            result['confidence'] = float(may_2025_api_score)
            
            results.append(result)
            logging.info(f"Item {i+1}/{len(items)}: May2025API:{'Y' if may_2025_api_relevant else 'N'}({may_2025_api_score:.4f}) May2025Opt:{'Y' if may_2025_optimal_relevant else 'N'}({may_2025_optimal_score:.4f}) URLOpt:{'Y' if url_optimal_relevant else 'N'}({url_optimal_score:.4f})")

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
        model_manager.end_request()

def run():
    server_address = ('0.0.0.0', 9092)
    httpd = HTTPServer(server_address, Handler)
    logging.info(f"Server started on {server_address[0]}:{server_address[1]}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
