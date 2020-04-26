from index import run_script, train_naive_bayes, train_logistic, train_tree

data1 = input('Enter data-set path for training. (Ex: file.txt): ')
data2 = input('Enter data-set path for training: ')
chosen_model = input('Enter model name for training. Options: naiveBayes, logisticRegression, treeClassifier: ')

tweets, vectors, additional_data = run_script(data1, data2)

if chosen_model == 'naiveBayes':
    print("Started training...")
    train_naive_bayes(additional_data)
elif chosen_model == 'logisticRegression':
    print("Started training...")
    train_logistic(vectors)
elif chosen_model == 'treeClassifier':
    print("Started training...")
    train_tree(vectors)
else:
    print('Model not found! Please only use one of these: naiveBayes, logisticRegression, treeClassifier')
