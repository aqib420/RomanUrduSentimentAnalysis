import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report, accuracy_score
from datasets import load_from_disk
from collections import Counter

# Load the fixed test dataset
combined_test = load_from_disk('data/combined_test')
X_test = combined_test['text']
y_test = combined_test['label']

# Load the augmented training dataset to recreate the tokenizer
combined_train_augmented = load_from_disk('data/combined_train_augmented')

# Recreate and fit the tokenizer using the augmented training dataset
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(combined_train_augmented['text'])

# Tokenize and pad the test dataset
X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=128)

# Load the trained LSTM model
lstm_model = tf.keras.models.load_model('models/lstm_sentiment_model.keras')

# Make predictions
y_pred_lstm = lstm_model.predict(X_test_padded).argmax(axis=1)

# Verify the class distributions in predictions and true labels
print("Unique classes in y_test:", set(y_test))
print("Unique classes in y_pred_lstm:", set(y_pred_lstm))

# Evaluate the LSTM model
print("LSTM Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_lstm))
print(classification_report(y_test, y_pred_lstm, target_names=['Negative', 'Positive', 'Neutral', 'Abusive']))
