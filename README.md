# Reddit Post Classifier

This project is a simple classifier for Reddit posts. It uses a pre-trained model to classify posts as relevant or irrelevant.

Created by [16x Tracker](https://tracker.16x.engineer/)

![screenshot](screenshots/screenshot.png)

## System Requirements

- Minimum 1GB RAM (2GB recommended)
- Docker and Docker Compose installed
- About 500MB disk space for the model and dependencies

## Sample Results

Apr 5 run

Pre-processing

```
Total entries processed: 9353
Unique entries: 4549
Duplicate entries: 1191
F5Bot filtered entries: 8
Team ID filtered entries (not team 1): 10359

Status breakdown:
RELEVANT/REPLIED: 241
IGNORED: 4246
NEW: 3
CONTENT_REMOVED: 59
```

Model Results

```
               precision    recall  f1-score   support

           0       0.98      0.95      0.96      1437
           1       0.25      0.50      0.33        50

    accuracy                           0.93      1487
   macro avg       0.62      0.72      0.65      1487
weighted avg       0.96      0.93      0.94      1487
```

Apr 18 run

Pre-processing

```
Total entries processed: 8473
Unique entries: 4258
Duplicate entries: 1174
F5Bot filtered entries: 8
Team ID filtered entries (not team 1): 0

Status breakdown:
RELEVANT/REPLIED: 274
IGNORED: 3828
NEW: 86
CONTENT_REMOVED: 70
```

Model Results

```
              precision    recall  f1-score   support

           0       0.97      0.94      0.95       766
           1       0.40      0.53      0.46        55

    accuracy                           0.92       821
   macro avg       0.68      0.74      0.71       821
weighted avg       0.93      0.92      0.92       821
```

## Running the API Server

### Using Docker Compose (Recommended)

The easiest way to run the service is using Docker Compose. The service will run in a container named `reddit-classifier-api`:

```bash
# pull latest changes from repo, rebuild the image and start the service
git pull && docker compose up --build -d

# view logs
docker compose logs -f

# Stop the service
docker compose down

# view container logs directly (using container name)
docker logs -f reddit-classifier-api
```

### Using Docker

Build the Docker image:

```bash
docker build -t reddit-post-classifier .
```

Run the container:

```bash
docker run -p 8080:8080 reddit-post-classifier
```

### Without Docker

Run the API server directly:

```bash
python api-server.py
```

## Making API Requests

### Single Post Classification

Make a request to classify a single post:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"content":"Your reddit post content goes here."}' \
  http://localhost:8080

# {"relevant": false, "confidence": 0.9925305247306824}

curl -X POST -H "Content-Type: application/json" \
  -d '{"content":"putting the question into a file, scheduling the launch, open project, paste the question and have Claude write the answer in a file"}' \
  http://localhost:8080

# {"relevant": true, "confidence": 0.6180585026741028}
```

### Bulk Post Classification

You can also classify multiple posts at once (up to 10 posts per request):

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"contents": [
    "Your first reddit post content goes here.",
    "Your second reddit post content goes here.",
    "Your third reddit post content goes here."
  ]}' \
  http://localhost:8080

# {
#   "results": [
#     {"relevant": false, "confidence": 0.9925305247306824},
#     {"relevant": true, "confidence": 0.6180585026741028},
#     {"relevant": false, "confidence": 0.9876543210987654}
#   ]
# }
```

If you submit more than 10 posts, the API will return a 400 error with a message indicating the maximum batch size has been exceeded.
