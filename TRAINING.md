# Training New Models with Fresh Data

## Prerequisites

Before training, ensure you have the required dependencies installed:

```bash
# Install training dependencies
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

## Data Preparation

### From CSV Export (Recommended)

The project includes an automated data preparation script for processing CSV exports from tracking systems:

```bash
python convert_hits_to_classifier_files.py
```

**CSV Requirements**:
- File should be named: `"Supabase Snippet Filter Tracking Hits for Team 1 May.csv"` (or update path in script)
- Required columns: `content`, `url`, `status`, `team_id`, `timestamp`
- Status values: `RELEVANT`, `REPLIED` (→ relevant_posts), `IGNORED` (→ irrelevant_posts)

**Configuration** (edit in script):
- `team_id_filter`: Filter by team ID (default: 1)
- `timestamp_threshold_days`: Filter posts older than N days (default: 90)
- Automatically removes F5Bot entries and duplicates

**Output**:
- Creates `relevant_posts/` and `irrelevant_posts/` directories
- Each file: `{content_hash}.txt` with URL on first line, content below

### Manual Data Format

If preparing data manually, organize in two folders:
- `relevant_posts/` - Posts classified as relevant  
- `irrelevant_posts/` - Posts classified as irrelevant

Each `.txt` file format:
```
https://reddit.com/r/example/comments/123/post_title/

This is the Reddit post content.
Multiple lines are supported.
```

## Training the Model

The current production model uses the **URL-Aware Regressor** approach:

```bash
python reddit-url-relevance-regressor.py
```

**Key Configuration** (edit in script):
- `RUN_NUMBER`: Increment for each training run (line 15)
- `PROJECT_NAME`: WandB project name (line 22) 
- Training defaults: 10 epochs, batch size 8, learning rate 2e-5

**Model Features**:
- RoBERTa-base with custom regression head
- URL-aware training (includes URL context)
- MSE loss with sigmoid activation
- Automatic optimal threshold finding
- Early stopping and checkpointing
- WandB experiment tracking

## Training Process

1. **Data Loading**: Loads from `relevant_posts/` and `irrelevant_posts/`
2. **Train/Validation Split**: 80/20 with stratification
3. **Model Training**: 
   - Early stopping (patience: 3 epochs)
   - Learning rate scheduling
   - Progress logging every 10 batches
4. **Model Saving**: Best model saved as `best_url_regressor_run{N}_epoch_{E}.pt`
5. **Threshold Optimization**: Finds optimal classification threshold (typically ~0.15)

## Model Evaluation

Test your trained model:

```bash
python test_regressor_apr.py
```

Expected performance on balanced test set:
- Overall accuracy: ~81-82%
- F1 Score: ~0.83
- Optimal threshold: ~0.15

## Key Hyperparameters

**Training Configuration**:
- `epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size (default: 8)
- `learning_rate`: AdamW learning rate (default: 2e-5)
- `patience`: Early stopping patience (default: 3)

**Model Configuration**:
- `max_length`: Token sequence length (default: 512)
- Base model: `roberta-base`

## Best Practices

1. **Data Quality**:
   - Use the CSV preprocessing script for consistent formatting
   - Ensure balanced relevant/irrelevant ratio (aim for 20-30% relevant)
   - Remove spam and low-quality content

2. **Training Monitoring**:
   - Monitor validation MSE and R² scores
   - Use WandB dashboard for experiment tracking
   - Early stopping prevents overfitting

3. **Model Updates**:
   - Increment `RUN_NUMBER` for each training experiment
   - Keep best models from different runs for comparison
   - Test multiple threshold values for production use

## Deployment

After training, update the production system:

1. Copy the best model file (e.g., `best_url_regressor_run1_epoch_5.pt`) to project root
2. Update `api-server.py` model path if needed
3. Rebuild and deploy:
```bash
docker compose up --build -d
```

The current production model (`best_url_regressor_run1_epoch_5.pt`) uses threshold 0.15 and achieves 81.67% accuracy.