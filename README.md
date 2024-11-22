# DistilBERT-based-sentiment-analysis-model
Features
Sentiment Classification: Fine-tuned DistilBERT model to classify WallStreetBets posts.
Auto-Labeling Pipeline: Utilized pre-trained models to generate sentiment labels for raw text data.
Model Fine-Tuning: Adapted DistilBERT for domain-specific language to improve classification accuracy.
Visualization Tools: Included scripts to generate sentiment trend graphs and confusion matrices for evaluation.
Side-by-Side Comparison: Provides comparisons between auto-labeled sentiments and fine-tuned predictions.

Usage
1. Auto-Labeling the Data
Run the SentimentIndicator.py script to process and auto-label the dataset:

bash
Copy code
python SentimentIndicator.py
Output: A CSV file (reddit_wsb_classified.csv) with sentiment labels.

2. Fine-Tuning the Model
Run the fine_tuning.py script to train and fine-tune the DistilBERT model:

bash
Copy code
python fine_tuning.py
Output:

Fine-tuned model saved in ./model/.
Training logs in ./logs/.
3. Visualizing Results
Run the visualize.py script to generate evaluation metrics and visualizations:

bash
Copy code
python visualize.py
Output: Graphs and metrics comparing auto-labeled and fine-tuned predictions, saved in the ./output/ directory.

File Structure
python
Copy code
├── data/
│   ├── reddit_wsb.csv               # Raw dataset
│   ├── reddit_wsb_classified.csv    # Auto-labeled dataset
├── model/
│   ├── fine_tuned_model.bin         # Fine-tuned DistilBERT model
├── logs/
│   ├── training_logs.json           # Training logs
├── output/
│   ├── comparison_results.csv       # Side-by-side comparison of sentiments
│   ├── graphs/                      # Visualizations
├── SentimentIndicator.py            # Script for auto-labeling the dataset
├── fine_tuning.py                   # Script for fine-tuning the model
├── visualize.py                     # Script for generating visualizations
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
Dataset
The dataset consists of Reddit posts scraped from the WallStreetBets subreddit.
Columns in reddit_wsb.csv:
body: The post content.
sentiment_classification: Auto-labeled sentiment (Positive, Neutral, Negative).
Outputs
Classification Report: Metrics like precision, recall, and F1-score for sentiment classes.
Confusion Matrix: Visualization of misclassifications.
Side-by-Side Comparison: CSV file showing differences between auto-labeled and fine-tuned predictions.
