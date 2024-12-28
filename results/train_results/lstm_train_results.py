import matplotlib.pyplot as plt

# Results from my LSTM run for training.
epochs = [1, 2, 3, 4, 5]
train_accuracy = [58.31, 73.60, 75.88, 78.10, 80.32]
val_accuracy = [67.54, 68.39, 68.39, 68.61, 68.43]
train_loss = [0.9281, 0.6202, 0.5619, 0.5103, 0.4623]
val_loss = [0.7114, 0.7064, 0.7172, 0.7482, 0.7702]

# Plot Accuracy
plt.figure(figsize=(5, 3))
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.title('LSTM Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss
plt.figure(figsize=(5, 3))
plt.plot(epochs, train_loss, label='Training Loss', marker='o', linestyle='--')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linestyle='--')
plt.title('LSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
