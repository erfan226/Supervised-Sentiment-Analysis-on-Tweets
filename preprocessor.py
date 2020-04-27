from __future__ import unicode_literals
import sys  # Remove later
import re
from hazm import *
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


class Preprocessor:
    """
    Preprocessor Core.
    """
    def normalizer(self, data: list):
        """
        Normalizes the data with Hazm library.

        :param list data: Data to be normalized
        :return: Normalized data
        """
        normalizer = Normalizer()
        normalized_data = []
        for item in data:
            normalized_item = normalizer.normalize(item)
            normalized_data.append(normalized_item)
        return normalized_data

    def tokenizer(self, data: list):
        """
        Tokenizes the data with Hazm library

        :param list data: Data to be tokenized
        :return: Tokenized list of documents
        """
        tokenized_list = []
        for item in data:
            tokens = word_tokenize(item)
            tokenized_list.append(tokens)
        return tokenized_list

    def remove_stop_words(self, data: list, stops: list, lemmatize: bool):
        """
        Removes stopwords and filters the data.

        :param list data: Data to be cleaned
        :param list stops: A list of stopwords to check on with and remove them from data
        :param bool lemmatize: If True, will also lemmatize tokens with Hazm library. Better not to use!
        :return: Cleaned data
        """
        output = []
        pattern = '[^\u0600-\u06FF#\r]'  # Remove numbers?
        for tokens in data:
            filtered_words = []
            if lemmatize:
                lemmatizer = Lemmatizer()
                stemmer = Stemmer()
                for token in tokens:
                    if token not in stops and len(re.findall(pattern, token)) <= 0:
                        if token.startswith("#"):
                            filtered_words.append(token)
                        else:
                            token = lemmatizer.lemmatize(token)
                            if token.find("#") != -1 and not token.startswith("#"):
                                filtered_words.append(token.split('#')[1])
                            else:
                                filtered_words.append(token)
                output.append(filtered_words)
            else:
                for token in tokens:
                    if token not in stops and len(re.findall(pattern, token)) <= 0:
                        filtered_words.append(token)
                output.append(filtered_words)
        return output

    # Below functions are not used and will be removed soon.
    def tokens_to_sentences(self, data: list):
        new_data = []
        for item in data:
            sent = ' '.join(item)
            new_data.append([sent])
        return new_data

    def get_ngram(self, data: list, n: int = 2):
        sentences_list = []
        ngram_data = []
        for item in data:
            for i in item:
                sentences_list.append(i)
            ngram = zip(*[sentences_list[j:] for j in range(n)])
            ngram_data.append([" ".join(ngram) for ngram in ngram])
        # x = [sentences_list[i:] for i in range(2)]
        return ngram_data

    def list_words(self, data: list):
        # To be removed
        # Convert tokens to vector
        # Maybe we should use NGrams for more accuracy and to keep word order
        list_of_words = []
        for item in data:
            for i, j in enumerate(item):
                if j not in list_of_words:
                    list_of_words.append(j)
        return list_of_words

    def check_entries(self, data: list, entries: list):
        word_checklist = []
        for entry in entries:
            items = re.findall(r"\(([^)]+)\)", entry)
            if len(items) == 1:
                item = items[0].replace("\"", "").split(",")
                word_checklist.append(item)
        for item in data:
            for i in item:
                for x in word_checklist:
                    if len(x) >= 2 and i in x:
                        print(i, "=>", x[2])
