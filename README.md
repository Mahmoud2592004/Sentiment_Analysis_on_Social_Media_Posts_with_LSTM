# Sentiment Analysis on Social Media Posts with LSTM

This project focuses on performing sentiment analysis on social media posts using an LSTM (Long Short-Term Memory) model. The goal is to classify the sentiment of posts (positive or negative) based on their content.

## Project Overview

The project involves building and evaluating a deep learning model using LSTM to capture the temporal dependencies in textual data. By using Word2Vec embeddings, the model translates words into numerical vectors that the LSTM network can process.

### Dataset

The dataset used for training and testing the model includes millions of social media posts. However, due to size limitations on GitHub, the dataset is not included in the repository. You can download it from [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) or use any other dataset for sentiment analysis.

### Steps Involved:

1. **Data Preprocessing**:
    - Removed noise from the text, including stopwords, special characters, and punctuation.
    - Tokenized the text to prepare it for vectorization.
    - Applied Word2Vec to convert text data into numerical vectors.

2. **Model Architecture**:
    - The model is built using an LSTM layer, designed to capture sequential dependencies in the data.
    - Hyperparameters:
        - Batch size: 128
        - Epochs: 5
        - LSTM hidden units: 8
        - Number of dense hidden units: 32
    - The model was trained using cross-entropy loss and Adam optimizer.

3. **Model Training and Validation**:
    - Initially, the model was trained without regularization, achieving training accuracy of 86.33% but a validation accuracy of only 77.20%, indicating overfitting.
    - To address overfitting, regularization (weight decay) was introduced. After several trials, the optimal regularization term was set to `0.000004`, resulting in an improved validation accuracy of 80.80%.

4. **Final Model Evaluation**:
    - The final model was evaluated on a test set using accuracy, F1-score, and confusion matrix.
    - Results:
        - Accuracy: 80.92%
        - F1-Score: 80.93%
    - The confusion matrix and classification report showed that the model performed equally well on both classes (positive and negative sentiments).

### Model Performance

The model achieved an accuracy of **80.92%** on the test set. Although the performance is good, there is room for improvement, especially in fine-tuning the model to handle more complex sentiment patterns (e.g., sarcasm or ambiguous sentiment).

### Potential Improvements:

1. **Data Augmentation**: Using more diverse datasets or data augmentation techniques to help the model generalize better.
2. **More Complex Architectures**: Experimenting with deeper or bidirectional LSTM models.
3. **Hyperparameter Tuning**: Further tuning of hyperparameters such as learning rate, batch size, and the number of LSTM units.

### How to Run the Project:

1. Clone this repository:
   ```bash
   git clone https://github.com/Mahmoud2592004/Sentiment_Analysis_on_Social_Media_Posts_with_LSTM.git
2. Install the required dependencies
3. Train the model: Run the provided Jupyter notebook to preprocess the data, build, and train the LSTM model.
4. Evaluate the model: The evaluation code in the notebook includes the generation of a confusion matrix and a detailed classification report.

