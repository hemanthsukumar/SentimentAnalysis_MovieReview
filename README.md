# Sentiment Analysis on Movie Reviews

## Project Overview
This project, developed as part of the **CSE-6363-004** course, focuses on performing sentiment analysis on movie reviews. The goal is to classify movie reviews as positive or negative using machine learning techniques. The dataset used is the **IMDb movie reviews dataset**, and the project implements data preprocessing, model training, and evaluation using two primary models: Convolutional Neural Networks (CNN) and Bidirectional Encoder Representations from Transformers (BERT).

### Team Members
- **Varun Perumandla** -  
- **Hemanth Sukumar Vangala** - 
- **Sathwik Reddy Avula** -

---

## Project Structure
The project involves the following key components:
1. **Data Preprocessing**
2. **Model Implementation**
3. **Model Training and Evaluation**
4. **Results and Analysis**

---

## Dataset
The dataset used in this project is the **IMDb movie reviews dataset** (`aclImdb`), which contains labeled movie reviews categorized as positive (`pos`) or negative (`neg`). The dataset is split into training and testing sets, with paths defined as:
- Training data: `aclImdb/train`
- Testing data: `aclImdb/test`

### Dataset Splitting
The dataset is split into training, validation, and testing sets as follows:
- **80%** of the data is used for training.
- **20%** of the data is used for testing.
- A further split of the training data is performed to create a validation set (80% training, 20% validation).

---

## Data Preprocessing
The preprocessing steps applied to the dataset include:
1. **File Reading and Dataframe Creation**:
   - Text files are read from the dataset directory and organized into a pandas DataFrame with columns `text` (review text) and `label` (sentiment: `pos` or `neg`).
   - Code snippet for this step:
     ```python
     import os
     import pandas as pd

     def read_text_files(directory, sentiment):
         data = []
         for file in os.listdir(directory):
             if file.endswith('.txt'):
                 with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                     data.append((f.read(), sentiment))
         return data

     def create_dataframe(data_directory):
         combined_data = []
         for sentiment in ['pos', 'neg']:
             sentiment_dir = os.path.join(data_directory, sentiment)
             combined_data.extend(read_text_files(sentiment_dir, sentiment))
         return pd.DataFrame(combined_data, columns=['text', 'label'])
     ```

2. **Text Cleaning**:
   - Removal of stopwords, punctuation, and HTML elements.
   - Conversion of text to lowercase.

3. **Lemmatization**:
   - Lemmatization is applied to normalize words and interpret the tone of sentences, aiding in sentiment analysis.

4. **Tokenization**:
   - Text data is tokenized to convert it into a numerical format suitable for machine learning models. This step enables the model to process the large volume of text data effectively.

---

## Models Used
Two models were implemented for sentiment analysis:

### 1. Convolutional Neural Network (CNN)
- **Purpose**: Achieves high accuracy in sentiment classification.
- **Implementation**:
  - A CNN model is built using a series of layers, including embedding, convolutional, dropout, and dense layers.
  - Code snippet for the CNN model:
    ```python
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=max_features, 
                                        output_dim=output_dim, 
                                        input_length=max_input_length))
    model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    ```

- **Training**:
  - The model is trained over 7 epochs, with training and validation accuracy/loss metrics recorded.
  - Example training output:
    ```
    Epoch 1/7: accuracy: 0.5014, val_accuracy: 0.6345
    Epoch 7/7: accuracy: 0.9313, val_accuracy: 0.8947
    ```

- **Results**:
  - Final accuracy on the validation set: **90%**.
  - Precision, recall, and F1-score for both classes (positive and negative) are approximately **0.88–0.89**.

### 2. Bidirectional Encoder Representations from Transformers (BERT)
- **Purpose**: Handles ambiguous messages and predicts sentiment effectively.
- **Implementation**:
  - BERT is implemented using the `ktrain` library, a lightweight wrapper for Keras.
  - Code snippet for BERT model setup:
    ```python
    model = text.text_classifier('bert', train_data=(X_train, y_train), preproc=preproc)
    learner = ktrain.get_learner(model=model, 
                                 train_data=train_data, 
                                 val_data=val_data, 
                                 batch_size=6)
    ```

- **Training**:
  - BERT is trained using the `onecycle` policy with a maximum learning rate of `2e-05`.

---

## Results
### CNN Model
- **Accuracy**: 90% on the validation set.
- **Performance Metrics**:
  | Class | Precision | Recall | F1-Score | Support |
  |-------|-----------|--------|----------|---------|
  | 0 (Negative) | 0.89 | 0.87 | 0.88 | 2463 |
  | 1 (Positive) | 0.88 | 0.89 | 0.89 | 2537 |
  | **Macro Avg** | 0.88 | 0.88 | 0.88 | 5000 |
  | **Weighted Avg** | 0.88 | 0.88 | 0.88 | 5000 |

- **Graphical Representation**:
  - Plots of model loss and accuracy are provided to visualize training progress.

### BERT Model
- Specific results for the BERT model are not detailed in the document but are expected to be competitive with or superior to the CNN model due to BERT's ability to handle ambiguous and complex text.

---

## Dependencies
To run this project, ensure the following libraries are installed:
- Python 3.x
- TensorFlow
- Keras
- pandas
- scikit-learn
- ktrain (for BERT implementation)
- nltk (for text preprocessing, including lemmatization)

You can install the required dependencies using:
```bash
pip install tensorflow pandas scikit-learn ktrain nltk
```

---

## How to Run the Project
1. Clone this repository to your local machine:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <repository-name>
   ```
3. Ensure the IMDb dataset (`aclImdb`) is placed in the project directory or update the paths in the code accordingly.
4. Run the preprocessing and model training scripts (ensure dependencies are installed):
   ```bash
   python <script-name>.py
   ```

---

## References
1. Dang, L., Wang, C., Han, H., & Hou, Y.-E. (2022). *A Hybrid BiLSTM-ATT Model for Improved Accuracy Sentiment Analysis*. [Link](https://ieeexplore-ieee-org.ezproxy.uta.edu/document/10074634)
2. Charitha, N. S. L. S., Rakesh, V., & Varun, M. (2023). *Comparative Study of Algorithms for Sentiment Analysis on IMDB Movie Reviews*. [Link](https://ieeexplore-ieee-org.ezproxy.uta.edu/document/10113113)
3. Mishra, M., & Patil, A. (2023). *Sentiment Prediction of IMDb Movie Reviews Using CNN-LSTM Approach*. [Link](https://ieeexplore-ieee-org.ezproxy.uta.edu/document/10165155)
4. *ktrain: A Lightweight Wrapper for Keras to Help Train Neural Networks*. [Link](https://towardsdatascience.com/ktrain-a-lightweight-wrapper-for-keras-to-help-train-neural-networks-82851ba889c)
5. *BERT Text Classification in 3 Lines of Code Using Keras*. [Link](https://towardsdatascience.com/bert-text-classification-in-3-lines-of-code-using-keras-264db7e7a358)

---
