
class Vectorizer:
    def feature_vector(self, data):
        """
        Takes the word tokens of documents and extracts a list of features from them.
        :param list data: List of lists that have word tokens
        :return: Extracted features of given documents
        """
        all_words = []
        for items in data:
            all_words.extend(items)
        return all_words

    def vecotrizer(self, features: list, row: list):
        """
        Takes a list of word tokens and converts them to number vectors according to features.
        :param list features: Extracted features
        :param list row: A list of binary representation of features
        :return: A list of converted document
        """
        vector = []
        for item in features:
            if item in row:
                vector.append(1)
            else:
                vector.append(0)
        return vector

    def token_to_vector(self, features, plus_cls, minus_cls):
        """
        Converts documents to vectors and labels each document.
        :param list features: Extracted features
        :param list plus_cls: Tokens of positive class documents
        :param list minus_cls: Tokens of negative class documents
        :return: A tuple which has vectors of both classes
        """
        plus_vectors = []
        for row in plus_cls:
            vec = self.vecotrizer(features, row)
            vec.insert(len(vec), 0)
            plus_vectors.append(vec)
        minus_vectors = []
        for row in minus_cls:
            vec = self.vecotrizer(features, row)
            vec.insert(len(vec), 1)
            minus_vectors.append(vec)
        return plus_vectors, minus_vectors

    def test_data_token_to_vector(self, features, test_data):
        """
        Converts the given test documents to vectors.
        :param list features: Extracted features
        :param list test_data: Document to convert
        :return: Vector representation of document
        """
        test_vector = []
        for row in test_data:
            vec = self.vecotrizer(features, row)
            test_vector.append(vec)
        return test_vector
