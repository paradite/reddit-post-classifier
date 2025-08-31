curl -X POST -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "content": "How I evaluate RTX 4080 against 3060 Ti",
        "url": "https://www.reddit.com/r/graphics_card/post/123"
      },
      {
        "content": "How to handle context for Claude Code, avoid copy-paste.",
        "url": "https://www.reddit.com/r/locallama/post/456"
      }
    ]
  }' \
  http://localhost:9092