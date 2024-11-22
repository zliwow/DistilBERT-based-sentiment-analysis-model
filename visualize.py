import pandas as pd
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Paths to your model, data, and logs
model_path = './model/fine_tuned_model.bin'
data_path = r'C:\Users\Andrew\Desktop\NEU\5100 foundation for AI\Final\Reddit WallStreetBets Posts Sentiment Analysis\reddit_wsb_classified.csv'
logs_path = './model/training_logs.json'

# Step 1: Load Data and Tokenizer
data = pd.read_csv(data_path)
data_sample = data.sample(frac=0.2, random_state=42)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data_sample['body'].tolist(), 
    data_sample['sentiment_classification'].map({'Positive': 2, 'Neutral': 1, 'Negative': 0}).tolist(), 
    test_size=0.2, 
    random_state=42
)

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# If a GPU is available, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Batch Inference on Validation Data
batch_size = 16
preds = []

# Ensure gradients are off and process in batches
model.eval()
with torch.no_grad():
    for i in range(0, len(val_texts), batch_size):
        batch_texts = val_texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        preds.extend(batch_preds)

# Step 3: Generate Classification Report
print("Classification Report:\n", classification_report(val_labels, preds, target_names=['Negative', 'Neutral', 'Positive']))

# Step 4: Display Confusion Matrix
cm = confusion_matrix(val_labels, preds, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Fine-Tuned Model")
plt.show()

# Step 5: Plot Training and Validation Loss from Logs
with open(logs_path, 'r') as f:
    logs = json.load(f)

training_loss = [entry['loss'] for entry in logs if 'loss' in entry]
validation_loss = [entry['eval_loss'] for entry in logs if 'eval_loss' in entry]
epochs = range(1, len(training_loss) + 1)

plt.figure(figsize=(10, 5))
if len(training_loss) == len(validation_loss) and len(training_loss) > 1:
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()
elif len(validation_loss) == 1:
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs (Only One Validation Loss Recorded)')
    plt.legend()
    plt.show()
else:
    print("Warning: Insufficient or mismatched data for validation loss. Skipping validation loss plot.")
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# Step 6: Side-by-Side Comparison of Auto-Labeled and Fine-Tuned Sentiment Predictions
print("\nSide-by-Side Comparison of Auto-Labeled and Fine-Tuned Model Predictions:\n")

# Sample 100 random posts for side-by-side comparison
sample_size = 100
sample_data = data.sample(n=sample_size, random_state=42)

# Define sentiment map for readable labels
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Function to get fine-tuned prediction
def get_fine_tuned_prediction(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return sentiment_map[predicted_class]

# Collect comparison data
comparison_results = []
for _, row in sample_data.iterrows():
    post_text = row['body']
    original_sentiment = row['sentiment_classification']
    fine_tuned_sentiment = get_fine_tuned_prediction(post_text)
    
    comparison_results.append({
        "Post Text": post_text,
        "Auto-Labeled Sentiment": original_sentiment,
        "Fine-Tuned Sentiment": fine_tuned_sentiment
    })

# Display the comparison
comparison_df = pd.DataFrame(comparison_results)
print(comparison_df.to_string(index=False))

output_file_path = './comparison_results.csv'
comparison_df.to_csv(output_file_path, index=False)
