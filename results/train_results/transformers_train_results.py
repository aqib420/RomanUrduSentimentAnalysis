import matplotlib.pyplot as plt

# BERT Training Results
bert_epochs = [1, 2, 3]
bert_train_accuracy = [65.0, 70.2, 75.67]
bert_val_accuracy = [63.5, 64.8, 65.5]
bert_train_loss = [0.989, 0.844, 0.7669]
bert_val_loss = [0.801, 0.761, 0.7426]

# GPT-2 Training Results
gpt2_epochs = [1, 2, 3]
gpt2_train_accuracy = [67.0, 75.0, 85.38]
gpt2_val_accuracy = [66.0, 69.0, 70.24]
gpt2_train_loss = [0.6675, 0.5323, 0.4462]
gpt2_val_loss = [0.6850, 0.6724, 0.6910]

# Plot BERT Accuracy
plt.figure(figsize=(5,3))
plt.plot(bert_epochs, bert_train_accuracy, label='BERT Training Accuracy', marker='o')
plt.plot(bert_epochs, bert_val_accuracy, label='BERT Validation Accuracy', marker='o')
plt.title('BERT Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# Plot BERT Loss
plt.figure(figsize=(5,3))
plt.plot(bert_epochs, bert_train_loss, label='BERT Training Loss', marker='o', linestyle='--')
plt.plot(bert_epochs, bert_val_loss, label='BERT Validation Loss', marker='o', linestyle='--')
plt.title('BERT Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot GPT-2 Accuracy
plt.figure(figsize=(5,3))
plt.plot(gpt2_epochs, gpt2_train_accuracy, label='GPT-2 Training Accuracy', marker='o')
plt.plot(gpt2_epochs, gpt2_val_accuracy, label='GPT-2 Validation Accuracy', marker='o')
plt.title('GPT-2 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# Plot GPT-2 Loss
plt.figure(figsize=(5,3))
plt.plot(gpt2_epochs, gpt2_train_loss, label='GPT-2 Training Loss', marker='o', linestyle='--')
plt.plot(gpt2_epochs, gpt2_val_loss, label='GPT-2 Validation Loss', marker='o', linestyle='--')
plt.title('GPT-2 Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
