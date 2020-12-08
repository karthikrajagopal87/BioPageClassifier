import collections
from keras import layers
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import plot_model
from utils import *

# ----------------------------------------------------------------------
# ** NOTE **
# This model overfits the data, I haven't been able to figure out why!
# However, Tensorflow classifier is able to accurately detect bio urls
# ----------------------------------------------------------------------

nlp = spacy.load("en_core_web_sm")

FilenameCollection = collections.namedtuple('FilenameCollection', 'texts_path urls_path')

# Seed for numpy randomizer
SEED = 1337

# Keras parameters
# ---------------------------------------------------------
# The name of the saved dataframe (helps with caching)
KERAS_TRAINING_SET_NAME = "training-data/keras_training_set.plk"

# The trained model name
KERAS_MODEL_NAME = "bio-model"

# The filename in which the graphical picture of Keras NN will be saved
KERAS_MODEL_ARCH_PLOT_NAME = "{}-arch-plot.png".format(KERAS_MODEL_NAME)

# The filename in which the training history plot will be saved
KERAS_MODEL_HISTORY_PLOT_NAME = "{}-history-plot.png".format(KERAS_MODEL_NAME)

# Location of Glove.6B dataset (will be used for the embedding layer)
# Download http://nlp.stanford.edu/data/glove.6B.zip
GLOVE_6B_PATH = '.glove.6B/glove.6B.300d.txt'

# Maximum length of the sequence
MAX_LEN = 1000

# Embedding Dimension
EMBEDDING_DIM = 100

# The maximum number of words to keep based on word frequency
NUM_WORDS = 5000

# Number of epochs for training
EPOCH = 10

# Batch size
BATCH_SIZE = 128

# Specifies how much of the training data should be allocated for validation
TEST_SIZE = 0.25

# Training data parameters
# ---------------------------------------------------------
# Positive training data locations
POSITIVE_TRAINING_DATA_FILENAMES = FilenameCollection(
    texts_path="training-data/bios.txt", urls_path="training-data/urls.text")

# Negative training data locations
NEGATIVE_TRAINING_DATA_FILENAMES = FilenameCollection(
    texts_path="training-data/neg-texts.txt", urls_path="training-data/neg-urls.txt")

# For model consistency and reproducibility
np.random.seed(SEED)


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


def create_model():
    """
    Creates the model, trains it, and saves it
    :return:
    """
    print("Loading training data")
    df = create_training_data(
        KERAS_TRAINING_SET_NAME,
        POSITIVE_TRAINING_DATA_FILENAMES.texts_path,
        NEGATIVE_TRAINING_DATA_FILENAMES.texts_path)

    # Splitting the training data
    print("Splitting training data into train/test sets")
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        df.text.values, df.label.values, test_size=TEST_SIZE, random_state=1000)

    # Creating a tokenizer
    print("Tokenize training and test sets")
    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(sentences_train)
    x_train = tokenizer.texts_to_sequences(sentences_train)
    print("Maximum sentence length is {}".format(max([len(seq) for seq in x_train])))

    x_test = tokenizer.texts_to_sequences(sentences_test)
    x_train = pad_sequences(x_train, padding='post', maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, padding='post', maxlen=MAX_LEN)

    # Creating embedding matrix
    print("Create word embedding matrix from Glove6B")
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = create_embedding_matrix(GLOVE_6B_PATH, tokenizer.word_index, EMBEDDING_DIM)
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print("Embedding Ratio: ", nonzero_elements / vocab_size)

    print("Creating the model")
    model = Sequential()
    model.add(layers.Embedding(
        vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False, mask_zero=True))
    # model.add(layers.Conv1D(128, 5, activation='relu'))
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