from index import run_script, run_knn, run_logistic, run_naive_bayes, run_tree, run_evaluation
from utils import read_labels, show_estimation_utility
import constants

data1 = input('Enter data-set path for training. (Ex: file.txt): ')
data2 = input('Enter data-set path for training: ')
chosen_model = input('Enter model name for testing. '
                     'Options: KNN, naiveBayes, logisticRegression, treeClassifier, finalModel: ')
test_file = input('Enter test file: ')
test_labels_file = input('Enter test labels file: ')

tweets, data_vectors, test_vector = run_script(data1, data2, test_file)
test_labels = read_labels(test_labels_file)

if chosen_model == 'KNN':
    run_knn(data_vectors, test_vector)
    confusion_matrix, accuracy, precision_recall_fscore = run_evaluation(test_labels, constants.KNN_PREDICTION)
    show_estimation_utility(confusion_matrix, accuracy, precision_recall_fscore)
elif chosen_model == 'naiveBayes':
    run_naive_bayes(test_vector, constants.BAYES_MODEL)
    confusion_matrix, accuracy, precision_recall_fscore = run_evaluation(test_labels, constants.BAYES_PREDICTION)
    show_estimation_utility(confusion_matrix, accuracy, precision_recall_fscore)
elif chosen_model == 'logisticRegression':
    run_logistic(test_vector)
    confusion_matrix, accuracy, precision_recall_fscore = run_evaluation(test_labels, constants.LOGISTIC_PREDICTION)
    show_estimation_utility(confusion_matrix, accuracy, precision_recall_fscore)
elif chosen_model == 'treeClassifier':
    run_tree(test_vector)
    confusion_matrix, accuracy, precision_recall_fscore = run_evaluation(test_labels, constants.TREE_PREDICTION)
    show_estimation_utility(confusion_matrix, accuracy, precision_recall_fscore)
elif chosen_model == 'finalModel':
    print("Please wait for the results. It might take a while, but it will worth it! :)", sep="")
    run_naive_bayes(test_vector, constants.FINAL_MODEL)
    confusion_matrix, accuracy, precision_recall_fscore = run_evaluation(test_labels, constants.BAYES_PREDICTION)
    show_estimation_utility(confusion_matrix, accuracy, precision_recall_fscore)
else:
    print('Model not found! Please only use one of these: KNN, naiveBayes, logisticRegression, treeClassifier,'
          'or finalModel.')
