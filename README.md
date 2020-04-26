# Tweet Classifier

A simple class project to classify a sample data-set of 1200 tweets to either positive or negative class, based on the
type of words used in those tweet.


#### Path to training files:
All data are located in the data directory.<br>
PLUS_TRAINING_DATA = `data/processed_plus_data.txt`<br>
MINUS_TRAINING_DATA = `data/processed_minus_data.txt`

## Instructions:
First do a `pip install -r requirements.txt` to install the required modules.
- To train the models, run `run_training.py`. First you should enter path to training data with
`PLUS_TRAINING_DATA` and then with `MINUS_TRAINING_DATA`. After that you should select an algorithm
to train. Options are `naiveBayes`, `logisticRegression`, `treeClassifier`.
- To test the trained models, run `run_test.py`. For the first input enter `PLUS_TRAINING_DATA` and `MINUS_TRAINING_DATA`. Then choose a model from `KNN`, `naiveBayes`, `logisticRegression`, `treeClassifier`, `finalModel`. Then enter the path to the test data file like `data/test.txt`. At last, enter path to label file for test data. Sample files are provided in `data` directory.
- To evaluate models, run `run_estimation.py`. For evaluating, just enter `PLUS_TRAINING_DATA` and `MINUS_TRAINING_DATA` as parameters.