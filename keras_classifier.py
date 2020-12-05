import collections
import re
import keras
import nltk
import pandas
import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from tinydb import TinyDB
from urllib.parse import urlparse
from os import path

FilenameCollection = collections.namedtuple('FilenameCollection', 'texts_path urls_path')

# Seed for numpy randomizer
SEED = 1237

# Maximum length of the sequence
MAX_LEN = 200

# Embedding Dimension
EMBEDDING_DIM = 100

# The maximum number of words to keep based on word frequency
NUM_WORDS = 5000

# Keras parameters
# ---------------------------------------------------------
# The name of the saved dataframe (helps with caching)
KERAS_TRAINING_SET_NAME = "keras_training_set.plk"

# The trained model name
KERAS_MODEL_NAME = "keras-bio-model-seed-{}".format(SEED)

# The filename in which the graphical picture of Keras NN will be saved
KERAS_MODEL_ARCH_PLOT_NAME = "{}-arch-plot.png".format(KERAS_MODEL_NAME)

# The filename in which the training history plot will be saved
KERAS_MODEL_HISTORY_PLOT_NAME = "{}-history-plot.png".format(KERAS_MODEL_NAME)

# Location of Glove.6B dataset (will be used for the embedding layer)
GLOVE_6B_PATH = '/home/machine/Downloads/glove.6B/glove.6B.300d.txt'

# Number of epochs for training
EPOCH = 100

# Batch size
BATCH_SIZE = 128

# Specifies how much of the training data should be allocated for validation
TEST_SIZE = 0.25

# Training data parameters
# ---------------------------------------------------------
# Crawler database filename
CRAWLER_DB_FILENAME = "crawler/pages.json"

# Positive training data locations
POSITIVE_TRAINING_DATA_FILENAMES = FilenameCollection(
    texts_path="training-data/bios.txt", urls_path="training-data/urls.text")

# Negative training data locations
NEGATIVE_TRAINING_DATA_FILENAMES = FilenameCollection(
    texts_path="training-data/pages.txt", urls_path="training-data/pages_urls.txt")

# For model consistency and reproducibility
np.random.seed(SEED)

# Checks to make sure stopwords corpora is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))


def create_negative_training_set():
    """
    Creates negative datasets from the pages crawler downloaded
    :return:
    """
    db = TinyDB(CRAWLER_DB_FILENAME)
    with open(POSITIVE_TRAINING_DATA_FILENAMES.urls_path, 'r') as b:
        bio_urls = b.readlines()
    bio_urls = [u.strip() for u in bio_urls]

    with open(NEGATIVE_TRAINING_DATA_FILENAMES.texts_path, "w") as pages, \
            open(NEGATIVE_TRAINING_DATA_FILENAMES.urls_path, 'w') as urls:
        for entry in tqdm(db):
            u = urlparse(entry["file"])
            if path.exists(u.path) and entry["url"] not in bio_urls:
                print(u.path)
                urls.write("{}\n".format(entry["url"]))
                with open(u.path, "r") as f:
                    content = f.readlines()
                pages.write("{}\n".format(" ".join(content)))


def preprocess_text(sen):
    """
    Preprocesses the sentence:
        - Remove punctuations and numbers
        - Single character removal
        - Removing multiple spaces
    :param sen: sentence
    :return: preprocessed sentence
    """
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence_tokens = word_tokenize(sentence)
    tokens_without_sw = [word for word in sentence_tokens if word not in STOP_WORDS]
    return " ".join(tokens_without_sw)


def load(filename):
    """
    Loads a data file where each line is a document
    :param filename: path to the data file
    :return: a list where each item is a line
    """
    print("Loading {}".format(filename))
    with open(filename, 'r') as f:
        content = []
        for line in tqdm(f.readlines()):
            content.append(preprocess_text(line.strip().lower()))
    return content


def create_embedding_matrix(filepath, word_index, embedding_dim):
    """
    Creates an embedding matrix from a Glove.6B file
    :param filepath: path to the glove file
    :param word_index: tokenizer word index
    :param embedding_dim: embedding dimension
    :return: embedding matrix
    """
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    print("Loading {}".format(filepath))
    with open(filepath) as f:
        for line in tqdm(f):
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


def create_training_data():
    """
    Creates training data frame from positive and negative labels
    :return: a panda data frame
    """
    if path.exists(KERAS_TRAINING_SET_NAME):
        return pandas.read_pickle(KERAS_TRAINING_SET_NAME)

    pos = TextData(text_path=POSITIVE_TRAINING_DATA_FILENAMES.texts_path, label=1)
    neg = TextData(text_path=NEGATIVE_TRAINING_DATA_FILENAMES.texts_path, label=0)
    df = pandas.concat([pos.to_df(), neg.to_df()], ignore_index=True).sample(frac=1)
    pandas.to_pickle(df, KERAS_TRAINING_SET_NAME)
    return df


class TextData(object):
    """
    TextData represents a group of text data that have the same label
    """
    def __init__(self, text_path, label):
        self._df = pandas.DataFrame()
        self._df['text'] = load(text_path)
        self._df['label'] = [label] * len(self._df)

    def size(self):
        return len(self._df)

    def sample(self, num_samples):
        return self._df[:num_samples]

    def to_df(self):
        return self._df


def show_loss_accuracy(name, model, x_train, x_test, y_train, y_test):
    """
    Shows loss and accuracy rate for the given model
    :param name: name of the model
    :param model: the trained model
    :param x_train: x training data
    :param x_test: x test data
    :param y_train: y training labels
    :param y_test: y test labels
    :return:
    """
    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print("{} Training Accuracy: {:.4f}".format(name, accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("{} Testing Accuracy:  {:.4f}".format(name, accuracy))


def plot_history(history, plot_filename=None):
    """
    Plots model training history (model accuracy and model loss)
    :param plot_filename: if provided, the plot will be saved
    :param history: training history
    :return:
    """
    fig, axs = plt.subplots(2, 1)
    fig.suptitle('Model Accuracy/Loss Plots')
    # Plots model accuracy
    axs[0].plot(history.history['acc'])
    axs[0].plot(history.history['val_acc'])
    axs[0].title.set_text('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'test'], loc='lower right')

    # Plots model loss
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].title.set_text('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'test'], loc='upper right')

    fig.tight_layout()
    plt.show()

    if plot_filename:
        fig.savefig(plot_filename)


def create_model():
    """
    Creates the model, trains it, and saves it
    :return:
    """
    print("Loading training data")
    df = create_training_data()

    # Splitting the training data
    print("Splitting training data into train/test sets")
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        df.text.values, df.label.values, test_size=TEST_SIZE, random_state=1000)

    # Creating a tokenizer
    print("Tokenize training and test sets")
    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(sentences_train)
    x_train = tokenizer.texts_to_sequences(sentences_train)
    x_test = tokenizer.texts_to_sequences(sentences_test)
    x_train = pad_sequences(x_train, padding='post', maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, padding='post', maxlen=MAX_LEN)

    # Creating embedding matrix
    print("Create word embedding matrix from Glove6B")
    vocab_size = len(tokenizer.word_index) + 1
    # Download http://nlp.stanford.edu/data/glove.6B.zip
    embedding_matrix = create_embedding_matrix(GLOVE_6B_PATH, tokenizer.word_index, EMBEDDING_DIM)
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print("Embedding Ratio: ", nonzero_elements / vocab_size)

    print("Creating the model")
    model = Sequential()
    model.add(layers.Embedding(
        vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.summary()

    print("Saving model architecture to {}".format(KERAS_MODEL_ARCH_PLOT_NAME))
    plot_model(model, to_file=KERAS_MODEL_ARCH_PLOT_NAME, show_shapes=True, show_layer_names=True)

    # Training the model
    history = model.fit(x_train, y_train, epochs=EPOCH, verbose=True, validation_data=(x_test, y_test),
                        batch_size=BATCH_SIZE)

    # Validation
    show_loss_accuracy("Original", model, x_train, x_test, y_train, y_test)

    # Plot
    plot_history(history, plot_filename=KERAS_MODEL_HISTORY_PLOT_NAME)

    # Saving the model
    model.save(KERAS_MODEL_NAME)

    # Asserting model works
    saved_model = keras.models.load_model(KERAS_MODEL_NAME)
    show_loss_accuracy("Saved", saved_model, x_train, x_test, y_train, y_test)
    return model


# Entry point
create_model()