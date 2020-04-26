from preprocessor import Preprocessor
import pickle
import random
from sklearn import metrics
import constants

prep = Preprocessor()


def file_reader(path: str = '', sub_list: bool = False):
    """
    Opens the given file and converts its content to a list
    :param str path: Path to the file
    :param bool sub_list:
    :return: Separated lines in a list
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
    list_data = file_reader(data)
    stop_words = file_reader(stops)
    return list_data, stop_words


def data_preprocessing(data1, data2, stops):
    # Normalizing the data
    plus_tweets_data = prep.normalizer(data1)
    minus_tweets_data = prep.normalizer(data2)
    # Tokenizing the data
    tokenized_plus_data = prep.tokenizer(plus_tweets_data)
    tokenized_minus_data = prep.tokenizer(minus_tweets_data)
    # Cleaning the data and maybe lemming the tokens
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

# Preprocessing test data
def test_data_preprocessing(test_data, stops):
    normal_test_data = prep.normalizer(test_data)
    tokenized_test_data = prep.tokenizer(normal_test_data)
    cleaned_test_data = prep.remove_stop_words(tokenized_test_data, stops, False)
    cleaned_test_data = [' '.join(data) for data in cleaned_test_data]
    return cleaned_test_data


# Save the given model and load them in pickle format
def save_model(file, model):
    print("Saving new model...")
    pkl_file = open(file, 'wb')
    pickle.dump(model, pkl_file)
    pkl_file.close()
    print("Model saved successfully!")


def load_model(file):
    pkl_file = open(file, 'rb')
    model = pickle.load(pkl_file)
    return model


# Takes all the vectors, separates labels and features and splits the data to train and test
def separate_labels(data):
    labels = []
    features = []
    for i in data:
        labels.append(i[-1])
        features.append(i[:-1])

    # x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0)
    # return x_train, x_test, y_train, y_test
    return features, labels


def special_split_data(data, split_value):
    random.shuffle(data)
    x_train = data[:split_value]
    x_test = data[:len(data)-split_value]
    y_test = data[:len(data)-split_value]
    y_test = [i[-1] for i in y_test]
    return x_train, x_test, y_test


def read_labels(path):
    """
    Read predicted labels from file.

    Parameters:
    path (str): Path to saved predictions file

    Returns:
    list: Returns a list of predicted labels
    """
    labels = []
    with open(path) as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels


def save_prediction(prediction, file_name):
    file = open(file_name, "w")
    for item in prediction:
        file.write(str(item)+"\n")
    file.close()


def estimate_accuracy(y_test, y_predict):
    accuracy = metrics.accuracy_score(y_test, y_predict)
    accuracy_percentage = 100 * accuracy
    return accuracy_percentage


def show_estimation_utility(confusion_matrix, accuracy, precision_recall_fscore):
    print("Confusion Matrix: \n", confusion_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision_recall_fscore[0])
    print("Recall:", precision_recall_fscore[1])
    print("F-Score:", precision_recall_fscore[2])
