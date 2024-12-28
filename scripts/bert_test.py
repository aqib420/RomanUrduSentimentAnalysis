from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from sklearn.metrics import classification_report, accuracy_score
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

# Load the updated combined test dataset
combined_test = load_from_disk('data/combined_test')
X_test = combined_test['text']
y_test = combined_test['label']

# Load the trained BERT model and tokenizer
bert_model_path = 'models\bert_sentiment_model'
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)

# Tokenize the test dataset
def tokenize_function(examples):
    return bert_tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

test_dataset = Dataset.from_dict({"text": X_test, "label": y_test})
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Use DataCollatorWithPadding for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer, return_tensors="pt")

# Prepare DataLoader
test_dataloader = DataLoader(
    tokenized_test.remove_columns(["text"]),
    batch_size=16,
    collate_fn=data_collator
)

# Evaluate the BERT model
bert_model.eval()
bert_model.to("cuda" if torch.cuda.is_available() else "cpu")

predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to("cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = batch['attention_mask'].to("cuda" if torch.cuda.is_available() else "cpu")
        labels = batch['labels'].to("cuda" if torch.cuda.is_available() else "cpu")

        # Perform the forward pass
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).tolist())
        true_labels.extend(labels.tolist())

# Print Evaluation Results
print("BERT Model Evaluation:")
print("Accuracy:", accuracy_score(true_labels, predictions))
print(classification_report(true_labels, predictions, target_names=['Negative', 'Positive', 'Neutral', 'Abusive']))
