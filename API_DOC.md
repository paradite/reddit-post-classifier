# Reddit Post Classifier API Documentation

This project is a simple classifier for Reddit posts. It uses a pre-trained model to classify posts as relevant or irrelevant.

## Making API Requests

The API provides results from three models:

1. Classifier model: Binary classification (relevant/not relevant) with confidence
2. Regressor model: Continuous score with threshold-based classification
3. URL Regressor model: Continuous score with threshold-based classification, using URL-prefixed text

### Single Post Classification

Make a request to classify a single post:

```bash
# Basic request (without URL)
curl -X POST -H "Content-Type: application/json" \
  -d '{"content":"Your reddit post content goes here."}' \
  http://localhost:8080
```

Response:

```json
{
  "classifier": {
    "relevant": false,
    "confidence": 0.9935186505317688
  },
  "regressor": {
    "score": 0.11482210457324982,
    "is_relevant": true,
    "threshold": 0.05
  },
  "relevant": false,
  "confidence": 0.9935186505317688
}
```

```bash
# Request with URL (for URL regressor model)
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "content": "Your reddit post content goes here.",
    "url": "https://www.reddit.com/r/example/post/123"
  }' \
  http://localhost:8080
```

Response:

```json
{
  "classifier": {
    "relevant": false,
    "confidence": 0.995292067527771
  },
  "regressor": {
    "score": 0.1207122877240181,
    "is_relevant": true,
    "threshold": 0.05
  },
  "url_regressor": {
    "score": 0.11297155916690826,
    "is_relevant": false,
    "threshold": 0.15
  },
  "relevant": false,
  "confidence": 0.995292067527771
}
```

### Bulk Post Classification

You can classify multiple posts at once (up to 10 posts per request). The API supports two formats:

1. Array of dictionaries (recommended):

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "content": "Your first reddit post content goes here.",
        "url": "https://www.reddit.com/r/example/post/123"
      },
      {
        "content": "Your second reddit post content goes here.",
        "url": "https://www.reddit.com/r/example/post/456"
      }
    ]
  }' \
  http://localhost:8080
```

Response:

```json
{
  "results": [
    {
      "classifier": {
        "relevant": false,
        "confidence": 0.9935186505317688
      },
      "regressor": {
        "score": 0.11482210457324982,
        "is_relevant": true,
        "threshold": 0.05
      },
      "url_regressor": {
        "score": 0.12649774551391602,
        "is_relevant": false,
        "threshold": 0.15
      },
      "relevant": false,
      "confidence": 0.9935186505317688
    },
    {
      "classifier": {
        "relevant": false,
        "confidence": 0.9967260360717773
      },
      "regressor": {
        "score": 0.07994376122951508,
        "is_relevant": true,
        "threshold": 0.05
      },
      "url_regressor": {
        "score": 0.10774697363376617,
        "is_relevant": false,
        "threshold": 0.15
      },
      "relevant": false,
      "confidence": 0.9967260360717773
    }
  ]
}
```

2. Separate arrays (legacy format):

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "Your first reddit post content goes here.",
      "Your second reddit post content goes here."
    ],
    "urls": [
      "https://www.reddit.com/r/example/post/123",
      "https://www.reddit.com/r/example/post/456"
    ]
  }' \
  http://localhost:8080
```

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {"content": "test1"}, {"content": "test2"}, {"content": "test3"},
      {"content": "test4"}, {"content": "test5"}, {"content": "test6"},
      {"content": "test7"}, {"content": "test8"}, {"content": "test9"},
      {"content": "test10"}
    ]
  }' \
  http://localhost:8080
```

Response format is identical to the array of dictionaries format

If you submit more than 10 posts, the API will return a 400 error:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {"content": "test1"}, {"content": "test2"}, {"content": "test3"},
      {"content": "test4"}, {"content": "test5"}, {"content": "test6"},
      {"content": "test7"}, {"content": "test8"}, {"content": "test9"},
      {"content": "test10"}, {"content": "test11"}
    ]
  }' \
  http://localhost:8080
```

Response:

```json
{
  "error": "Request exceeded maximum batch size of 10. Received 11 items."
}
```

## Understanding the Response Fields

1. **Classifier Model**:

   - `relevant`: Boolean indicating if the post is classified as relevant (true) or not relevant (false)
   - `confidence`: Confidence score of the classification (0.0 to 1.0)

2. **Regressor Model**:

   - `score`: Continuous relevance score (0.0 to 1.0)
   - `is_relevant`: Boolean indicating if the score is above the threshold (0.05)
   - `threshold`: The threshold value used for classification (0.05)

3. **URL Regressor Model** (only included when URL is provided):

   - `score`: Continuous relevance score (0.0 to 1.0)
   - `is_relevant`: Boolean indicating if the score is above the threshold (0.15)
   - `threshold`: The threshold value used for classification (0.15)

4. **Legacy Fields** (for backward compatibility):
   - `relevant`: Same as classifier.relevant
   - `confidence`: Same as classifier.confidence

Note: The URL regressor model processes the text with the URL prefixed (e.g., "https://www.reddit.com/r/example/post/123\n\nYour post content"), while the classifier and regressor models process the text without the URL prefix. This matches how each model was trained.
