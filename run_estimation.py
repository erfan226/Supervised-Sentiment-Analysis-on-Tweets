from index import Bayes_k_fold, Logistic_k_fold, Tree_k_fold, KNN_k_fold, run_script
import numpy as np

data1 = input('Enter data-set path for training. (Ex: file.txt): ')
data2 = input('Enter data-set path for training: ')

tweets, vectors, additional_data = run_script(data1, data2)

# Removes labels from vectors
x = [i[:-1] for i in vectors]
y = []
for i in vectors:
    y.append(i[-1])
features = np.array(x)
labels = np.array(y)

print("\nBayes model running...")
Bayes_k_fold(features, labels, additional_data)
print("\nLogistic model running...")
Logistic_k_fold(features, labels)
print("\nTree model running...")
Tree_k_fold(features, labels)
print("\nKNN model running...This one takes a long time!")
KNN_k_fold(features, labels)
