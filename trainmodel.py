import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight

# Constants for text processing
MAX_VOCAB = 10000
MAX_LENGTH = 120  # Adjusted based on your dataset analysis

def load_data(file_path):
    """
    Load the preprocessed data from a CSV file.
    """
    return pd.read_csv(file_path)

def create_tokenizer(texts, max_vocab=MAX_VOCAB):
    """
    Fits a tokenizer on the texts.
    """
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def tokenize_and_pad(texts, tokenizer, max_length=MAX_LENGTH):
    """
    Tokenizes and pads the text data using the given tokenizer.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences

def prepare_input_data(train_data_path, test_data_path):
    # Load saved preprocessed data
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)

    # Convert to string and fill NaN values with a placeholder text like 'missing'
    train_data['cleaned_text'] = train_data['cleaned_text'].fillna('missing').astype(str)
    test_data['cleaned_text'] = test_data['cleaned_text'].fillna('missing').astype(str)

    # Create a tokenizer and fit it on the training data
    tokenizer = create_tokenizer(train_data['cleaned_text'])

    # Tokenize and pad the text data for both training and test sets
    X_train = tokenize_and_pad(train_data['cleaned_text'], tokenizer)
    X_test = tokenize_and_pad(test_data['cleaned_text'], tokenizer)
    
    y_train = train_data['sentiment'].values
    y_test = test_data['sentiment'].values
    
    return X_train, y_train, X_test, y_test, tokenizer.word_index

def compute_class_weights(labels):
    """
    Computes class weights to handle imbalanced datasets.
    """
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))
