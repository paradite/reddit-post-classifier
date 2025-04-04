# Reddit Post Classifier

This project is a simple classifier for Reddit posts. It uses a pre-trained model to classify posts as relevant or irrelevant.

Created by [16x Tracker](https://tracker.16x.engineer/)

## Results

### polar-wave-5

- Trained for 3 epochs

5746015ef3a8bc2661ef1db0e135539847d113b5

```
               precision    recall  f1-score   support

           0       0.97      0.99      0.98      1437
           1       0.50      0.26      0.34        50

    accuracy                           0.97      1487
   macro avg       0.74      0.63      0.66      1487
weighted avg       0.96      0.97      0.96      1487
```

### ruby-vortex-6

- Trained for 10 epochs
- WeightedRandomSampler

b894522593fdae0422b3c0c2a4ac9c8dd30f2bf3

```
               precision    recall  f1-score   support

           0       0.98      0.95      0.96      1437
           1       0.25      0.50      0.33        50

    accuracy                           0.93      1487
   macro avg       0.62      0.72      0.65      1487
weighted avg       0.96      0.93      0.94      1487
```

## Running the API Server

### Using Docker Compose (Recommended)

The easiest way to run the service is using Docker Compose. The service will run in a container named `reddit-classifier-api`:

```bash
# Start the service
docker compose up -d

# View logs
docker compose logs -f

# Stop the service
docker compose down

# View container logs directly (using container name)
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

Make a request to the API:

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
