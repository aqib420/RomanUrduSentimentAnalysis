import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_from_disk

# Load augmented training and validation datasets
combined_train = load_from_disk('data/combined_train_augmented')
combined_validation = load_from_disk('data/combined_validation')

# Tokenization and Padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(combined_train['text'])  # Fit tokenizer on training data

# Training Data
X_train = pad_sequences(tokenizer.texts_to_sequences(combined_train['text']), maxlen=128)
y_train = tf.keras.utils.to_categorical(combined_train['label'], num_classes=4)

# Validation Data
X_val = pad_sequences(tokenizer.texts_to_sequences(combined_validation['text']), maxlen=128)
y_val = tf.keras.utils.to_categorical(combined_validation['label'], num_classes=4)

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=128),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model with Validation Data
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=5
)

# Save Model
model.save('models/lstm_sentiment_model.keras')
print("LSTM Model Saved in .keras format")
