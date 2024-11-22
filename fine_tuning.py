import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load the classified data
data_path = r'C:\Users\Andrew\Desktop\NEU\5100 foundation for AI\Final\Reddit WallStreetBets Posts Sentiment Analysis\reddit_wsb_classified.csv'
data = pd.read_csv(data_path)

# Sample 20% of the dataset for faster training
data_sample = data.sample(frac=0.2, random_state=42)

# Split sampled data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data_sample['body'].tolist(), 
    data_sample['sentiment_classification'].map({'Positive': 2, 'Neutral': 1, 'Negative': 0}).tolist(), 
    test_size=0.2, 
    random_state=42
)

# Load pre-trained tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Add labels to the encodings
train_encodings['labels'] = train_labels
val_encodings['labels'] = val_labels

# Convert encodings to Dataset format expected by the Trainer
train_dataset = Dataset.from_dict(train_encodings)
val_dataset = Dataset.from_dict(val_encodings)

# Define Trainer arguments and training parameters with optimizations
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,      
    per_device_eval_batch_size=4,
    num_train_epochs=1,                
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True                           # Mixed precision for faster training on GPU
)

# Define Trainer with loss and accuracy logging
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the fine-tuned model and training logs
model.save_pretrained('./model/fine_tuned_model.bin')
with open('./model/training_logs.json', 'w') as f:
    json.dump(trainer.state.log_history, f)
