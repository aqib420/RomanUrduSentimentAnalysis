# Roman Urdu Sentiment Analysis
This project implements sentiment analysis for Roman Urdu using LSTM, BERT, and GPT-2. Below is the folder structure:

- `data/`: Contains the datasets (before and after augmentation).
- `models/`: Includes trained models in appropriate formats.
- `scripts/`: Python scripts for training, testing, and data preparation.
- `results/`: Training and testing results along with visualizations.

## How to Run the Project
1. Install dependencies using `requirements.txt` by running this command in the terminal->"pip install -r requirements.txt".
2. Run `eda.py` to prepare datasets followed by `augment_abusive.py` to augment the imbalanced class.
3. Execute training scripts (`lstm_train.py`, etc.).
4. Evaluate models with respective testing scripts and based on the results check the graphs for testing results.

## Data
The `data/` folder contains datasets and processed files. Due to size constraints, this folder is not included in the repository. You can generate or download the required files using the following steps:

- To preprocess the data, run:
  ```bash
  python scripts/EDA.py



#   R o m a n U r d u S e n t i m e n t A n a l y s i s 
 
 
