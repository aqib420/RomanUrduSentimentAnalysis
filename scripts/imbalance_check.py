from datasets import load_from_disk
from collections import Counter

# Load datasets
combined_train = load_from_disk("data/combined_train")
combined_validation = load_from_disk("data/combined_validation")
combined_test = load_from_disk("data/combined_test")

# Count labels for each dataset
def count_labels(dataset, label_name):
    label_counts = Counter(dataset[label_name])
    return label_counts

# Combined dataset counts
train_counts = count_labels(combined_train, 'label')
val_counts = count_labels(combined_validation, 'label')
test_counts = count_labels(combined_test, 'label')

print(f"Training Dataset Class Distribution: {train_counts}")
print(f"Validation Dataset Class Distribution: {val_counts}")
print(f"Test Dataset Class Distribution: {test_counts}")

# Specifically check the count for abusive/offensive class (label=3)
abusive_train = train_counts.get(3, 0)
abusive_val = val_counts.get(3, 0)
abusive_test = test_counts.get(3, 0)

print(f"Abusive Class Counts: Train = {abusive_train}, Validation = {abusive_val}, Test = {abusive_test}")

