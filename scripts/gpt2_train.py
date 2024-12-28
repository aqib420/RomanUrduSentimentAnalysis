from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

# Load training and validation datasets
combined_train = load_from_disk('data/combined_train_augmented')
combined_validation = load_from_disk('data/combined_validation')

# Load GPT-2 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad token to the same as eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize Datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

train_dataset = combined_train.map(tokenize_function, batched=True)
val_dataset = combined_validation.map(tokenize_function, batched=True)

# Load Pre-trained GPT-2 Model
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=4)

# Explicitly set the padding token for the model
model.config.pad_token_id = tokenizer.pad_token_id

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results_gpt2',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=1,
    logging_dir='./logs_gpt2'
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

# Save the GPT-2 model
model.save_pretrained('models/gpt2_sentiment_model')
tokenizer.save_pretrained('models/gpt2_sentiment_model')

print("GPT-2 Model and Tokenizer Saved!")
