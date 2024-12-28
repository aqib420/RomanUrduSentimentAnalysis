import numpy as np

classes = ['Negative', 'Positive', 'Neutral', 'Abusive']
lstm_f1 = [51, 69, 63, 31]
bert_f1 = [57, 78, 69, 42]
gpt2_f1 = [65, 82, 75, 40]

x = np.arange(len(classes))
width = 0.25

plt.bar(x - width, lstm_f1, width, label='LSTM')
plt.bar(x, bert_f1, width, label='BERT')
plt.bar(x + width, gpt2_f1, width, label='GPT-2')

plt.xlabel('Classes')
plt.ylabel('F1-Score (%)')
plt.title('Class-wise F1-Score Comparison')
plt.xticks(x, classes)
plt.legend()
plt.show()
