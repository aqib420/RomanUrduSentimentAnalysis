import matplotlib.pyplot as plt

models = ['LSTM', 'BERT', 'GPT-2']
accuracy = [57.43, 65.05, 70.24] #my test run results final accuracies for all the models

plt.bar(models, accuracy, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Overall Accuracy Comparison')
plt.ylim(0, 100)
plt.show()
