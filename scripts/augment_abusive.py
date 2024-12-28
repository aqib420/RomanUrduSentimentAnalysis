import random
from collections import Counter
from nltk import ngrams
from datasets import Dataset, concatenate_datasets
from datasets import load_from_disk
import re

# Load datasets
combined_train = load_from_disk("data/combined_train")
combined_validation = load_from_disk("data/combined_validation")
combined_test = load_from_disk("data/combined_test")

# Define stopwords specific to Roman Urdu
stopwords = set([
    "hai", "ki", "ko", "k", "ka", "to", "ho", "ye", "se", "ke", "tu", "aur", 
    "bhi", "is", "or", "hain", "me", "ha", "nahi", "bc", "h", "hi", "na", "kar", 
    "tum", "he", "nhi", "kya", "jo", "kr", "tha", "koi", "ne", "apni", "and", 
    "ab", "the", "tere", "mein", "a", "pe", "rt"
])

# Define synonym dictionary in Roman Urdu
synonym_dict = {
    "randi": ["randwa", "kanjri", "bazari aurat"],
    "bhenchod": ["bhnchod", "bhan ke lode", "behn kay lovde"],
    "madarchod": ["madrchod", "maa kay lode", "maa ka ch**"],
    "teri maa": ["teri ami", "teray maa", "teri wali maa"],
    "chod": ["kar", "maar", "lut"],
    "lanat": ["sharam", "ghalat kaam", "baddua"],
    "harami": ["kameena", "kamina", "ghaleez"],
    "ganda": ["mela", "kachra", "ghaleez"],
    "beghairat": ["beshram", "besharam", "namard"],
    "maa choot": ["maa ka gali", "ami ka ankh", "maa ka sharafat"],
    "yahoodi agent": ["yahoodi ka chamcha", "israel ka agent", "agent yahoodi"]
}

# Function to get word frequencies
def get_word_frequencies(dataset):
    word_counts = Counter()
    for sample in dataset:
        if sample['text']:  # Skip None or empty texts
            words = re.findall(r'\b\w+\b', sample['text'].lower())  # Tokenize text
            filtered_words = [word for word in words if word not in stopwords]
            word_counts.update(filtered_words)
    return word_counts

# Function for synonym replacement
def synonym_replacement(text, synonym_dict, num_replacements=2):
    words = text.split()
    replaced_words = words.copy()
    count = 0
    
    for i, word in enumerate(words):
        if word in synonym_dict and count < num_replacements:
            replaced_words[i] = random.choice(synonym_dict[word])
            count += 1
    return " ".join(replaced_words)

# Function to augment dataset
def augment_dataset(dataset, synonym_dict, augment_factor=3):
    augmented_texts = []
    for i in range(len(dataset)):
        text = dataset[i]['text']
        if text:  # Skip None or empty texts
            for _ in range(augment_factor):  # Augment each sample multiple times
                augmented_texts.append(synonym_replacement(text, synonym_dict))
    
    # Create an augmented dataset
    augmented_dataset = Dataset.from_dict({
        'text': augmented_texts,
        'label': [3] * len(augmented_texts)  # Assign label 3 to all augmented samples
    })
    return augmented_dataset

# Filter the abusive class (label = 3) from the training dataset
abusive_samples = combined_train.filter(lambda example: example['label'] == 3)

# Remove None or empty texts
abusive_samples = abusive_samples.filter(lambda example: example['text'] is not None)

# Analyze the dataset for common abusive terms
word_frequencies = get_word_frequencies(abusive_samples)
print("Most Common Abusive Unigrams:", word_frequencies.most_common(20))

# Augment the abusive class samples using synonym replacement
augmented_abusive_samples = augment_dataset(abusive_samples, synonym_dict, augment_factor=3)
print(f"Number of Augmented Abusive Samples: {len(augmented_abusive_samples)}")

# Combine augmented samples with the original training dataset
combined_train = concatenate_datasets([combined_train, augmented_abusive_samples])

# Verify the updated class distribution
train_counts = Counter(combined_train['label'])
print(f"Updated Training Dataset Class Distribution: {train_counts}")

# Save the augmented training dataset
combined_train.save_to_disk("data/combined_train_augmented")
print("Augmented training dataset saved successfully!")
