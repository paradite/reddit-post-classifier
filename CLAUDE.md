# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reddit Post Classifier is a machine learning project that classifies Reddit posts as relevant/irrelevant using three different RoBERTa-based transformer models. The project includes training scripts, model evaluation tools, and a production API server.

## Essential Commands

### Docker Operations (Primary Development/Deployment)

```bash
# Deploy API server
docker compose up --build -d

# View logs
docker compose logs -f

# Stop service
docker compose down
```

### Testing

```bash
# Test local API server
./scripts/test-api.sh

# Test remote API server
./scripts/test-remote-api.sh

# Evaluate model performance
python test_model.py

# Test regressor models
python test_regressor_apr.py
```

### Training

For more details, see @TRAINING.md

```bash
# Train main classifier
python reddit-semantic-classifier.py

# Train regressor models
python reddit-relevance-regressor.py
python reddit-url-relevance-regressor.py
```

## Architecture Overview

### Three-Model System

1. **Classifier Model** (`best_model_run12_epoch_9.pt`) - Binary classification with confidence
2. **Regressor Model** (`best_regressor_run1_epoch_4.pt`) - Continuous scoring (threshold: 0.05)
3. **URL Regressor Model** (`best_url_regressor_run1_epoch_5.pt`) - URL-aware scoring (threshold: 0.15)

### Key Components

- **api-server.py** - HTTP server (port 9092) with dynamic model loading
- **Training data** - `relevant_posts/` (~200 files), `irrelevant_posts/` (~800+ files)
- **Model artifacts** - Pre-trained PyTorch models (.pt files)
- **Docker containerization** - Memory-limited deployment (3.5GB limit)

### Data Format

Training files contain URL on first line, followed by post content. API expects:

```json
{
  "contents": [
    {
      "content": "Reddit post text",
      "url": "https://reddit.com/r/example/post/123"
    }
  ]
}
```

## Dependencies

- **Runtime**: torch>=2.0.0+cpu, transformers>=4.28.0, psutil>=5.9.0
- **Training**: pandas==2.2.3, scikit-learn==1.3.2, wandb==0.19.9
- **Python**: Fixed at 3.11.10
- **Package manager**: uv (preferred) with fallback to pip

## Performance Notes

- Models achieve 80-96% accuracy on test data
- API processes up to 10 posts per request
- Memory management includes automatic model loading/unloading
- WandB integration for experiment tracking
