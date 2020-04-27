from preprocessor import Preprocessor
import pickle
from sklearn import metrics
import constants

prep = Preprocessor()


def file_reader(path: str = '', sub_list: bool = False):
    """
    Opens the given file and converts its content to a list.

    :param str path: Path to the file
    :param bool sub_list: Converts each line of the document into separated word list
    :return: Separated lines in a list
    :raises FileNotFoundError: If file was not found.
    """
    if path != '':
        lines = []
        err_code = 0
        err = 'No file with path "{}" was found'
        try:
            with open(path) as data:
                for line in data:
                    if sub_list:
                        lines.append(line.strip().split())
                    else:
                        lines.append(line.strip())
            return lines
        except FileNotFoundError:
            return err_code, err.format(path)
    else:
        return ''


def read_raw_data(data, stops):
    """
    Converts test document and stopwords to a list.

    :param str data: Test document file
    :param str stops: Stopwords file
    :return: Converted files to a tuple of lists
    """
    list_data = file_reader(data)
    stop_words = file_reader(stops)
    return list_data, stop_words


def data_preprocessing(data1, data2, stops):
    """Main preprocessor function.

    Takes in the documents and stopwords and normalizes the documents, tokenizes them, removes the stopwords and
    saves them as a file.

    :param list data1: Positive class documents
    :param list data2: Negative class documents
    :param list stops: Stopwords
    :return: None
    """
    plus_tweets_data = prep.normalizer(data1)
    minus_tweets_data = prep.normalizer(data2)
    tokenized_plus_data = prep.tokenizer(plus_tweets_data)
    tokenized_minus_data = prep.tokenizer(minus_tweets_data)
    cleaned_plus_tweets = prep.remove_stop_words(tokenized_plus_data, stops, False)
    cleaned_minus_tweets = prep.remove_stop_words(tokenized_minus_data, stops, False)
    cleaned_plus_tweets = [' '.join(data) for data in cleaned_plus_tweets]
    cleaned_minus_tweets = [' '.join(data) for data in cleaned_minus_tweets]
    f1 = open(constants.PROCESSED_PLUS_DATA, 'w')
    for line in cleaned_plus_tweets:
        f1.write(line+'\n')
    f1.close()
    f2 = open(constants.PROCESSED_MINUS_DATA, 'w')
    for line in cleaned_minus_tweets:
        f2.write(line+'\n')
    f2.close()


def test_data_preprocessing(test_data, stops):
    """Preprocessor function for test documents

    Takes in the test documents and stopwords and normalizes the documents, tokenizes them, removes the stopwords and
    returns them as a list.

    :param list test_data: Test documents
    :param list stops: Stopwords
    :return: Tokenized words of test document
    """
    normal_test_data = prep.normalizer(test_data)
    tokenized_test_data = prep.tokenizer(normal_test_data)
    cleaned_test_data = prep.remove_stop_words(tokenized_test_data, stops, False)
    cleaned_test_data = [' '.join(data) for data in cleaned_test_data]
    return cleaned_test_data


def save_model(file, model):
    """
    Takes in a model and saves it as a pickle file.

    :param str file: Path to the file
    :param model: Trained model object to be saved
    """
    print("Saving new model...")
    pkl_file = open(file, 'wb')
    pickle.dump(model, pkl_file)
    pkl_file.close()
    print("Model saved successfully!")


def load_model(file):
    """
    Takes in a model and saves it as a pickle file.

    :param str file: Path to the file
    :return: Trained model
    """
    pkl_file = open(file, 'rb')
    model = pickle.load(pkl_file)
    return model


def separate_labels(data):
    """
    Takes all the vectors, then separates labels and features and finally splits the data to train-set and test-set

    :param list data: Vector of documents
    :return: Vector of documents and their separated labels
    """
    labels = []
    features = []
    for i in data:
        labels.append(i[-1])
        features.append(i[:-1])

    return features, labels


# def special_split_data(data, split_value):
#     random.shuffle(data)
#     x_train = data[:split_value]
#     x_test = data[:len(data)-split_value]
#     y_test = data[:len(data)-split_value]
#     y_test = [i[-1] for i in y_test]
#     return x_train, x_test, y_test


def read_labels(path):
    """
    Read predicted labels from file and converts it to a list.

    :param str path: Path to saved predictions file
    :return: Predicted labels
    """
    labels = []
    with open(path) as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels


def save_prediction(prediction, file_name):
    """
    Saves the predicted values of a model to a text file.

    :param list prediction: A list of labels to be saved
    :param str file_name: Path to the file
    :return: None
    """
    file = open(file_name, "w")
    for item in prediction:
        file.write(str(item)+"\n")
    file.close()


def estimate_accuracy(y_test, y_predict):
    """
    Estimates the accuracy of the given model from predicted labels.

    :param list y_test: True labels of test documents
    :param list y_predict: Predicted labels of test document
    :return: Accuracy of the corresponding model
    """
    accuracy = metrics.accuracy_score(y_test, y_predict)
    accuracy_percentage = 100 * accuracy
    return accuracy_percentage


def show_estimation_utility(confusion_matrix, accuracy, precision_recall_fscore):
    """
    Prints the statistical information of model from the given input.

    :param numpy.ndarray confusion_matrix: Confusion matrix of the tested model
    :param float accuracy: Accuracy of the tested model
    :param tuple precision_recall_fscore: Statistical info of tested model
    :return: None
    """
    print("Confusion Matrix: \n", confusion_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision_recall_fscore[0])
    print("Recall:", precision_recall_fscore[1])
    print("F-Score:", precision_recall_fscore[2])
