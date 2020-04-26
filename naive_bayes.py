class NaiveBayes:
    """
    Naive Bayes Algorithm.
    Attributes
    ----------
    docs_prob: list
        Probability of each document.
    features_prob: list
        Probability of each feature.
    """
    def __init__(self):
        self.docs_prob = []
        self.features_prob = []

    def class_probability(self, *args):
        """
        Takes the whole document and assigns a probability to each document.
        :param list args: Documents of each class in the form of word tokens
        :return list: Each index holds a probability of the corresponding document
        """
        docs_length = []
        for item in args:
            docs_length.append(len(item))
        total_docs_len = sum(docs_length)
        for item in docs_length:
            self.docs_prob.append(item/total_docs_len)

    # Any numbers of classes can be used for estimation.
    # We will be using Laplace Smoothing to avoid zero-probability when the feature is not observed in a class
    # but in the other
    def feature_probability(self, features, *args):
        """
        Takes all of the features and for each class, then assigns a probability to it.
        These probabilities are accessible through features_prob attribute.
        :param list features: All of the extracted features
        :param list args: The probability of each class
        """
        for cls in args:
            temp_prob = []
            for item in features:
                word_count = cls.count(item)
                len_words = len(cls)
                word_prob = word_count / len_words
                if word_prob == 0:
                    # We can round it up/down but it may get too small than already is
                    word_prob = (word_count + 1) / (len_words + 1)
                temp_prob.append(word_prob)
            self.features_prob.append(temp_prob)

    def probability_calculation(self, test_vector, docs_prob, features_prob):
        """
        Takes a test vector and based on the probability of each feature for each class,
        multiplies them with the probability of their class, then calculates the probability of the given vector.
        :param list test_vector: Vector to be tested
        :param list docs_prob: Probability of all classes
        :param list features_prob: Probability of all feature
        :return int: Value of the class with the biggest probability
        """
        cls_prob = []
        predicted_class = 0
        # Avoid counting labels as features
        for i, doc_prob in enumerate(docs_prob):
            value = 1
            for j, feature in enumerate(test_vector[:-1]):
                if feature == 1:
                    value = features_prob[i][j] * value
            cls_prob.append(value * doc_prob)
            predicted_class = cls_prob.index(max(cls_prob))
            # Making sure of unseen data not causing problems
            # if 1 not in vector:
            #     value = 0
        return predicted_class

    # May result in floating-point underflow
    def predict_class(self, test_vectors, docs_prob, features_prob):
        """
        Takes test vectors and for each, calculates its probability and assigns a class to it.
        :param list test_vectors: List of all test vectors
        :param list docs_prob: Probability of all classes
        :param list features_prob: Probability of all feature
        :return list: Predicted class for each test vector
        """
        estimated_classes = []
        for vector in test_vectors:
            estimated_classes.append(self.probability_calculation(vector, docs_prob, features_prob))
        return estimated_classes
