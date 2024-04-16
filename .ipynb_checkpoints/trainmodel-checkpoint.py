import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight

# Constants for text processing
MAX_VOCAB = 10000  # Maximum vocabulary size
MAX_LENGTH = 120  # Maximum length of sequences

def load_data(file_path):
    """
    Load data from a CSV file into a DataFrame.
    """
    return pd.read_csv(file_path)

def create_tokenizer(texts, max_vocab=MAX_VOCAB):
    """
    Creates and fits a tokenizer on the given texts.
    """
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def tokenize_and_pad(texts, tokenizer, max_length=MAX_LENGTH):
    """
    Tokenizes and pads the text data.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences

def compute_class_weights(labels):
    """
    Computes class weights to balance the dataset.
    """
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))

def prepare_input_data(file_path):
    """
    Prepares input data for model training:
    - Loads data
    - Ensures all text entries are strings and handles NaN values.
    - Tokenizes and pads the sequences.
    - Computes class weights.
    Returns tokenized and padded text data, labels, word index from tokenizer, and class weights.
    """
    df = load_data(file_path)
    # Replace NaN values with an empty string to ensure text processing works correctly
    df['text'] = df['text'].fillna('')
    tokenizer = create_tokenizer(df['text'])
    X = tokenize_and_pad(df['text'], tokenizer)
    y = df['sentiment'].values
    class_weights = compute_class_weights(y)

    return X, y, tokenizer.word_index, class_weights
