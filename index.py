from vectorizer import Vectorizer
from knn import KNN
from naive_bayes import NaiveBayes
import utils
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import pandas as pd
import constants

make_vector = Vectorizer()
knn = KNN()
naive_bayes = NaiveBayes()
logistic_regression = LogisticRegression()  # penalty="l2", solver="newton-cg"
tree_classifier = tree.DecisionTreeClassifier(max_depth=50, max_features=420)  # criterion="entropy", max_depth=40, max_features=40


def run_script(data1, data2, test_data_file: str = None):
    """
    Executes as the entry program is executed to pass the data & vectors to other modules and functions.

    :param str data1: Path to training data file
    :param str data2: Path to training data file
    :param str test_data_file: Path to test file. Only set when testing & evaluating
    :return: Tokenized documents, vectors, feature vector and separated feature vectors
    """
    print("Processing...")
    # Preprocessed data-set are prepared for vector transformation
    pre_plus_data = utils.file_reader(data1, True)
    pre_minus_data = utils.file_reader(data2, True)
    tweets = pre_plus_data + pre_minus_data
    # Extracted features for each class
    plus_feature_vector = make_vector.feature_vector(pre_plus_data)
    minus_feature_vector = make_vector.feature_vector(pre_minus_data)
    # Tokens are merged which are used as the features. Also removing the duplicates
    temp_fv = plus_feature_vector + minus_feature_vector
    feature_vector = list(dict.fromkeys(temp_fv))
    # Tokens into vectors
    plus_vectors, minus_vectors = make_vector.token_to_vector(feature_vector, pre_plus_data, pre_minus_data)
    vectors = plus_vectors + minus_vectors
    # A list of necessary data to train Naive Bayes model
    additional_data = [feature_vector, plus_feature_vector, minus_feature_vector, pre_plus_data, pre_minus_data]
    # These operations are only executed in Test phase
    if test_data_file is not None:
        test_data, stops = utils.read_raw_data(test_data_file, constants.STOP_WORDS)
        pre_test_data = utils.test_data_preprocessing(test_data, stops)
        # Preprocessed data-set
        tokenized_test_data = [x.split() for x in pre_test_data]
        # Tokens into vectors
        test_vector = make_vector.test_data_token_to_vector(feature_vector, tokenized_test_data)
        return tweets, vectors, test_vector
    else:
        return tweets, vectors, additional_data


def run_knn(vectors, test_vectors, labels=None):
    """
    Runs the K Nearest Neighbors algorithm and saves the predicted values to a file.

    :param numpy.ndarray vectors: An array of sample vectors
    :param numpy.ndarray test_vectors: An array of test vectors
    :param list labels: Labels of test vectors to put it back in the vector
    :return: None
    """
    predictions = []
    # Due to KNN Structure we need to put the labels back to its vectors. Bummer!
    if labels is not None:
        vectors = vectors.tolist()
        test_vectors = test_vectors.tolist()
        for i, item in enumerate(vectors):
            item.insert(len(item), labels[i])
    for test_vec in test_vectors:
        predictions.append(knn.knn_predict(vectors, test_vec, 3))
    utils.save_prediction(predictions, constants.KNN_PREDICTION)


def run_naive_bayes(test_vectors, model):
    """
    Runs the Naive Bayes algorithm with a model and saves the predicted values.

    :param numpy.ndarray test_vectors: An array of test vectors
    :param model: Predict with the chosen model
    :return: None
    """
    naive_model = utils.load_model(model)
    predictions = naive_bayes.predict_class(test_vectors, naive_model.docs_prob, naive_model.features_prob)
    utils.save_prediction(predictions, constants.BAYES_PREDICTION)


def run_logistic(test_vectors):
    """
    Runs the Logistic Regression algorithm with a model and saves the predicted values.

    :param numpy.ndarray test_vectors: An array of test vectors
    :return: None
    """
    logistic_regression = utils.load_model(constants.LOGISTIC_MODEL)
    predictions = logistic_regression.predict(test_vectors)
    utils.save_prediction(predictions, constants.LOGISTIC_PREDICTION)


def run_tree(test_vectors):
    """
    Runs the Tree Classifier algorithm with a model and saves the predicted values.

    :param numpy.ndarray test_vectors: An array of test vectors
    :return: None
    """
    tree_classifier = utils.load_model(constants.TREE_MODEL)
    predictions = tree_classifier.predict(test_vectors)
    utils.save_prediction(predictions, constants.TREE_PREDICTION)


def train_naive_bayes(data):
    """
    Trains a model with Naive Bayes algorithm and save the model.\n
    First it will calculate the probability of each class according to the number of its documents. Then it calculates
    the probability of each feature from feature vector.

    :param list data: Includes feature vector and other information from run_script()
    :return: None
    """
    naive_bayes.class_probability(data[3], data[4])
    naive_bayes.feature_probability(data[0], data[1], data[2])
    utils.save_model(constants.BAYES_MODEL, naive_bayes)


def train_logistic(vectors):
    """
    Trains a model with Logistic Regression algorithm and save the model.\n
    Before saving, the labels must be separated from vectors.

    :param list vectors: Vectors of documents
    :return: None
    """
    x_train, y_train = utils.separate_labels(vectors)
    logistic_regression.fit(x_train, y_train)
    utils.save_model(constants.LOGISTIC_MODEL, logistic_regression)


def train_tree(vectors):
    """
     Trains a model with Tree Classifier algorithm and save the model.\n
     Before saving, the labels must be separated from vectors.

     :param list vectors: Vectors of documents
     :return: None
     """
    x_train, y_train = utils.separate_labels(vectors)
    tree_classifier.fit(x_train, y_train)
    utils.save_model(constants.TREE_MODEL, tree_classifier)


def run_evaluation(test_labels, predicted_labels):
    """
    Gives statistical information on the performance of given algorithm.

    :param list test_labels: Real value of test labels
    :param str predicted_labels: Predicted value of test labels
    :return: Confusion Matrix, Accuracy, Precision, Recall, fMeasure
    """
    predicted_labels = utils.read_labels(predicted_labels)
    co_matrix = confusion_matrix(test_labels, predicted_labels)
    accuracy = utils.estimate_accuracy(test_labels, predicted_labels)
    precision_recall_fscore = precision_recall_fscore_support(test_labels, predicted_labels, average='weighted')
    print(class_predictions(test_labels, predicted_labels))
    return co_matrix, accuracy, precision_recall_fscore


def class_predictions(test_labels, predicted_labels):
    """
    Shows prediction values for each class.

    :param list test_labels: Real value of test labels
    :param list predicted_labels: Predicted value of test labels
    :return: Prediction of all test documents for each class
    """
    t_labels = pd.Series(test_labels)
    p_labels = pd.Series(predicted_labels)
    table = pd.crosstab(t_labels, p_labels, rownames=['True'], colnames=['Predicted'], margins=True)
    return table


def KNN_k_fold(features, labels):
    """
    KFold estimation for K Nearest Neighbors algorithm. For k numbers it will split the data into train/test set
    and trains, tests and evaluates the model.

    :param numpy.ndarray features: Vector of documents
    :param numpy.ndarray labels: Label of vectors
    :return: None
    """
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        run_knn(x_train, x_test, y_train)
        co_matrix, accuracy, precision_recall_fscore = run_evaluation(y_test.tolist(), constants.KNN_PREDICTION)
        utils.show_estimation_utility(co_matrix, accuracy, precision_recall_fscore)


def Bayes_k_fold(features, labels, data: list):
    """
    KFold estimation for Naive Bayes algorithm. For k numbers it will split the data into train/test set
    and trains, tests and evaluates the model.

    :param numpy.ndarray features: Vector of documents
    :param numpy.ndarray labels: Label of vectors
    :param list data: Some information for bayes algorithm
    :return: None
    """
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        train_naive_bayes(data)
        run_naive_bayes(x_test, constants.BAYES_MODEL)
        co_matrix, accuracy, precision_recall_fscore = run_evaluation(y_test.tolist(), constants.BAYES_PREDICTION)
        utils.show_estimation_utility(co_matrix, accuracy, precision_recall_fscore)


def Logistic_k_fold(features, labels):
    """
    KFold estimation for Logistic Regression algorithm. For k numbers it will split the data into train/test set
    and trains, tests and evaluates the model.

    :param numpy.ndarray features: Vector of documents
    :param numpy.ndarray labels: Label of vectors
    :return: None
    """
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        logistic_regression.fit(x_train, y_train)
        utils.save_model(constants.LOGISTIC_MODEL, logistic_regression)
        run_logistic(x_test)
        co_matrix, accuracy, precision_recall_fscore = run_evaluation(y_test.tolist(), constants.LOGISTIC_PREDICTION)
        utils.show_estimation_utility(co_matrix, accuracy, precision_recall_fscore)


def Tree_k_fold(features, labels):
    """
    KFold estimation for Tree Classifier algorithm. For k numbers it will split the data into train/test set
    and trains, tests and evaluates the model.

    :param numpy.ndarray features: Vector of documents
    :param numpy.ndarray labels: Label of vectors
    :return: None
    """
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        tree_classifier.fit(x_train, y_train)
        utils.save_model(constants.TREE_MODEL, tree_classifier)
        run_tree(x_test)
        co_matrix, accuracy, precision_recall_fscore = run_evaluation(y_test.tolist(), constants.TREE_PREDICTION)
        utils.show_estimation_utility(co_matrix, accuracy, precision_recall_fscore)
