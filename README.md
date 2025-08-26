# NewsCategoryClassification

## Project Overview
This project implements a Vietnamese news article classification system using LSTM neural networks and Word2Vec embeddings. The system can categorize Vietnamese text into 18 different news categories.

## Dataset
The project uses the "hiuman/vietnamese_classification" dataset which contains Vietnamese news articles labeled with categories such as:
- Business (kinh_doanh)
- Law (pháp_luật)
- Education (giáo_dục)
- Technology (công_nghệ)
- And more...

## Project Structure
- `preprocessing.py` - Text preprocessing utilities for Vietnamese text
- `W2V.py` - Word2Vec model training for word embeddings
- `make_embedded_vector.py` - Converts text into embedding vectors
- `make_corpus.py` - Creates vocabulary corpus from the dataset
- `dataset.py` - PyTorch dataset wrapper for the text data
- `LSTM.py` - LSTM model definition and training script
- `test_model.py` - Model evaluation script
- `checkGPU.py` - Utility to check GPU availability
- `test_metrics.csv` - Contains evaluation metrics from the trained model

## Setup and Installation

### Requirements
```
torch
underthesea
gensim
numpy
pandas
scikit-learn
datasets
tqdm
nltk
```

### Running the Project

1. Check hardware compatibility:
```
python checkGPU.py
```

2. Create the Word2Vec model:
```
python W2V.py
```

3. Generate embeddings for the dataset:
```
python make_embedded_vector.py
```

4. Train the LSTM model:
```
python LSTM.py
```

5. Test model performance:
```
python test_model.py
```

## Model Architecture
The project uses a simple LSTM architecture:
- Bidirectional LSTM layer
- Dense layer with ReLU activation
- Dropout for regularization
- Output layer with softmax activation

## Results
The model achieves excellent classification performance across the 18 news categories. Detailed metrics are saved to `test_metrics.csv` after running the test script.

### Performance Metrics
From our latest evaluation (`test_metrics.csv`):
- Accuracy: 99.44%
- Precision: 99.44%
- Recall: 99.44%
- F1 Score: 99.44%

These high scores indicate that our LSTM model with Word2Vec embeddings performs extremely well on Vietnamese news classification tasks.

## Future Improvements
- Experiment with different embedding techniques (BERT, fastText)
- Implement data augmentation for imbalanced categories
- Try different model architectures (Transformer-based models)

- ## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
