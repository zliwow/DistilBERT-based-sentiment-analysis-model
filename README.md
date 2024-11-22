# DistilBERT-based-sentiment-analysis-model
**Features**
**Sentiment Classification**
Fine-tuned DistilBERT model to classify Reddit posts into Positive, Neutral, and Negative sentiments.

**Auto-Labeling Pipeline**
Automatically labeled raw Reddit posts using a pre-trained model to generate sentiment categories, forming the basis for fine-tuning.

**Model Fine-Tuning**
Adapted the pre-trained DistilBERT model for WallStreetBets-specific language and tone, improving classification accuracy.

**Visualization Tools**
Generated visualizations like confusion matrices, classification reports, and sentiment distribution charts to analyze and present results effectively.

**Side-by-Side Comparison**
Compared auto-labeled sentiments with fine-tuned model predictions, showcasing improvements and discrepancies.

**Usage**
1. Auto-Labeling the Data
Run the SentimentIndicator.py script to process and auto-label the dataset:

python SentimentIndicator.py
Output: A CSV file (reddit_wsb_classified.csv) with sentiment labels.

2. Fine-Tuning the Model
Run the fine_tuning.py script to train and fine-tune the DistilBERT model:

python fine_tuning.py
Output:
Fine-tuned model saved in ./model/.
Training logs in ./logs/.

3. Visualizing Results
Run the visualize.py script to generate evaluation metrics and visualizations:

python visualize.py
Output: Graphs and metrics comparing auto-labeled and fine-tuned predictions, saved in the ./output/ directory.


**Dataset**
The dataset consists of Reddit posts scraped from the WallStreetBets subreddit.
Columns in reddit_wsb.csv:
body: The post content.
sentiment_classification: Auto-labeled sentiment (Positive, Neutral, Negative).
Outputs
Classification Report: Metrics like precision, recall, and F1-score for sentiment classes.
Confusion Matrix: Visualization of misclassifications.
Side-by-Side Comparison: CSV file showing differences between auto-labeled and fine-tuned predictions.




