curl -X POST -vv -H "Content-Type: application/json" \
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
  http://64.23.229.254:9092