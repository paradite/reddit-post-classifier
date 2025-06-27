# Reddit Post Classifier API Documentation

This project is a simple classifier for Reddit posts. It uses pre-trained models to classify posts as relevant or irrelevant.

## Model Updates (June 2025)

**BREAKING CHANGE**: The API has been updated to serve only the top 3 performing models based on comprehensive evaluation results:

1. **URL Regressor May 2025 (API)** - F1: 0.9167 (Primary model)
2. **URL Regressor May 2025 (Optimal)** - F1: 0.8679 
3. **URL Regressor (Optimal)** - F1: 0.6964

**Removed models** (due to poor performance):
- ~~Classifier model~~ - F1: 0.4062 (removed)
- ~~Regressor model~~ - F1: 0.5763 (removed)

## Making API Requests

The API now provides results from the top 3 performing models, all of which are URL-aware regression models that use URL-prefixed text when URLs are provided.

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
  http://localhost:9092
```

Response:

```json
{
  "results": [
    {
      "url_regressor_may_2025_api": {
        "score": 0.8451,
        "is_relevant": true,
        "threshold": 0.15
      },
      "url_regressor_may_2025_optimal": {
        "score": 0.8451,
        "is_relevant": true,
        "threshold": 0.0342
      },
      "url_regressor_optimal": {
        "score": 0.7234,
        "is_relevant": true,
        "threshold": 0.1203
      },
      "relevant": true,
      "confidence": 0.8451
    },
    {
      "url_regressor_may_2025_api": {
        "score": 0.1245,
        "is_relevant": false,
        "threshold": 0.15
      },
      "url_regressor_may_2025_optimal": {
        "score": 0.1245,
        "is_relevant": true,
        "threshold": 0.0342
      },
      "url_regressor_optimal": {
        "score": 0.0987,
        "is_relevant": false,
        "threshold": 0.1203
      },
      "relevant": false,
      "confidence": 0.1245
    }
  ]
}
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
  http://localhost:9092
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
  http://localhost:9092
```

Response:

```json
{
  "error": "Request exceeded maximum batch size of 10. Received 11 items."
}
```

## Understanding the Response Fields

### Model Results

1. **url_regressor_may_2025_api** (Primary Model - F1: 0.9167):
   - `score`: Continuous relevance score (0.0 to 1.0)
   - `is_relevant`: Boolean indicating if the score is above the threshold (0.15)
   - `threshold`: The threshold value used for classification (0.15)

2. **url_regressor_may_2025_optimal** (Secondary Model - F1: 0.8679):
   - `score`: Continuous relevance score (0.0 to 1.0)
   - `is_relevant`: Boolean indicating if the score is above the threshold (0.0342)
   - `threshold`: The threshold value used for classification (0.0342)

3. **url_regressor_optimal** (Tertiary Model - F1: 0.6964):
   - `score`: Continuous relevance score (0.0 to 1.0)
   - `is_relevant`: Boolean indicating if the score is above the threshold (0.1203)
   - `threshold`: The threshold value used for classification (0.1203)

### Primary Response Fields

- `relevant`: Boolean from the best performing model (url_regressor_may_2025_api)
- `confidence`: Score from the best performing model (url_regressor_may_2025_api)

### Model Processing

All models process text with URL prefixed when URLs are provided (e.g., "https://www.reddit.com/r/example/post/123\n\nYour post content"). When no URL is provided, only the post content is processed. This URL-aware approach significantly improves classification accuracy.
