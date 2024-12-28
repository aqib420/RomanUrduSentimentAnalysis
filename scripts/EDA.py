from datasets import load_dataset, concatenate_datasets, Value, ClassLabel, Features
from collections import Counter

# Load the datasets
sentiment_dataset = load_dataset("HowMannyMore/romanurdu-sentiment-dataset")
hate_speech_dataset = load_dataset("community-datasets/roman_urdu_hate_speech")

# Convert ClassLabel to integer labels
def convert_classlabel_to_int(dataset):
    if isinstance(dataset.features['label'], ClassLabel):
        dataset = dataset.cast_column('label', Value('int64'))
    return dataset

for split in hate_speech_dataset:
    hate_speech_dataset[split] = convert_classlabel_to_int(hate_speech_dataset[split])

# Map sentiment dataset labels
def map_sentiment_labels(batch):
    label_mapping = {'Negative': 0, 'Positive': 1, 'Neutral': 2}
    batch['label'] = [label_mapping.get(label, -1) for label in batch['label']]
    return batch

sentiment_dataset = sentiment_dataset.map(map_sentiment_labels, batched=True)

# Map hate speech dataset labels with fallback for invalid labels
def map_hate_speech_labels(batch):
    label_mapping = {
        0: 3,  # 'Abusive/Offensive' -> 3
        1: 2   # 'Normal' -> 2 (Neutral)
    }
    # Update -1 labels to 3
    batch['label'] = [
        label_mapping.get(label, 3) if label is not None else 3
        for label in batch['label']
    ]
    return batch

hate_speech_dataset = hate_speech_dataset.map(map_hate_speech_labels, batched=True)

# Remove invalid labels (-1) from sentiment dataset
sentiment_dataset = sentiment_dataset.filter(lambda example: example['label'] != -1)

# Combine the datasets
combined_train = concatenate_datasets([sentiment_dataset['train'], hate_speech_dataset['train']])
combined_validation = concatenate_datasets([sentiment_dataset['valid'], hate_speech_dataset['validation']])
combined_test = concatenate_datasets([sentiment_dataset['test'], hate_speech_dataset['test']])

# Save the datasets
combined_train.save_to_disk("data/combined_train")
combined_validation.save_to_disk("data/combined_validation")
combined_test.save_to_disk("data/combined_test")

# Verify class distributions
print("Training Dataset Class Distribution:", Counter(combined_train['label']))
print("Validation Dataset Class Distribution:", Counter(combined_validation['label']))
print("Test Dataset Class Distribution:", Counter(combined_test['label']))
