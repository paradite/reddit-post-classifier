# In a new file, e.g., simple_api_server.py

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
MODEL_PATH = "reddit_topic_classifier.pt"

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
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            text = data.get('content', '')
            logging.info(f"Received request with text length: {len(text)}")
        except:
            logging.error("Failed to parse request data")
            self.send_response(400)
            self.end_headers()
            return

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

        response = {'relevant': bool(pred == 1), 'confidence': float(conf)}
        logging.info(f"Prediction: {'relevant' if pred == 1 else 'not relevant'} (confidence: {conf:.2f})")

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run():
    server_address = ('0.0.0.0', 8080)
    httpd = HTTPServer(server_address, Handler)
    logging.info(f"Server started on {server_address[0]}:{server_address[1]}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
