# In a new file, e.g., simple_api_server.py

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
MODEL_PATH = "reddit_topic_classifier_2.pt"
MAX_BATCH_SIZE = 10

# Load model
logging.info("Loading model and tokenizer...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

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
                texts = [data.get('content', '')]
            elif 'contents' in data:
                # Bulk content request
                texts = data.get('contents', [])
                if not isinstance(texts, list):
                    raise ValueError("'contents' must be a list")
                if len(texts) > MAX_BATCH_SIZE:
                    error_msg = f"Request exceeded maximum batch size of {MAX_BATCH_SIZE}. Received {len(texts)} items."
                    logging.error(error_msg)
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": error_msg}).encode('utf-8'))
                    return
            else:
                raise ValueError("Request must contain either 'content' or 'contents' field")
                
            logging.info(f"Received request with {len(texts)} text(s)")
        except Exception as e:
            logging.error(f"Failed to parse request data: {str(e)}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return

        results = []
        for i, text in enumerate(texts):
            text_start_time = time.time()
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

            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits
                probs = torch.softmax(logits, dim=1)
                pred = int(torch.argmax(probs, dim=1).item())
                conf = probs[0][pred].item()

            result = {'relevant': bool(pred == 1), 'confidence': float(conf)}
            results.append(result)
            text_time = time.time() - text_start_time
            logging.info(f"Text {i+1}/{len(texts)}: {'relevant' if pred == 1 else 'not relevant'} (confidence: {conf:.2f}, time: {text_time:.3f}s)")

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
