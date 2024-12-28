from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

# Load training and validation datasets
combined_train = load_from_disk('data/combined_train_augmented')
combined_validation = load_from_disk('data/combined_validation')

# Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize Datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

train_dataset = combined_train.map(tokenize_function, batched=True)
val_dataset = combined_validation.map(tokenize_function, batched=True)

# Load Pre-trained BERT Model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=1,
    logging_dir='./logs'
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train Model
trainer.train()

# Save the BERT model
model.save_pretrained('models/bert_sentiment_model')
tokenizer.save_pretrained('models/bert_sentiment_model')

print("BERT Model and Tokenizer Saved!")

