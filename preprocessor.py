from __future__ import unicode_literals
import sys  # Remove later
import re
from hazm import *
# from numpy import array, sqrt, reshape
import numpy as np
# import cProfile
from profilehooks import profile
from random import randrange

np.set_printoptions(threshold=sys.maxsize)

class Preprocessor:
    def normalizer(self, data: list):
        normalizer = Normalizer()
        normalized_data = []
        for item in data:
            normalized_item = normalizer.normalize(item)
            normalized_data.append(normalized_item)
        return normalized_data

    def tokenizer(self, data: list):
        tokenized_list = []
        for item in data:
            tokens = word_tokenize(item)
            tokenized_list.append(tokens)
        return tokenized_list

    def remove_stop_words(self, data: list, stops: list, lemmatize: bool):
        output = []
        pattern = '[^\u0600-\u06FF#\r]'  # Remove numbers?
        lemmatizer = Lemmatizer()
        # stemmer = Stemmer()
        i = 0
        for tokens in data:
            filtered_words = []
            if lemmatize:
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
                # filtered_words = [token for token in tokens if
                #               token not in stops and len(re.findall(pattern, token)) <= 0]
                for token in tokens:
                    if token not in stops and len(re.findall(pattern, token)) <= 0:
                        filtered_words.append(token)

                        # if token.startswith("#"):
                        #     # print(i, token)
                        #     filtered_words.append(token)
                        #     i += 1
                        # else:
                        #     filtered_words.append(token)
                output.append(filtered_words)
        return output

    def tokens_to_sentence(self, data: list):
        new_data = []
        for item in data:
            sent = ' '.join(item)
            new_data.append([sent])
        return new_data

    # @profile
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
        # Maybe we should use NGrams for more accuracy and to keep word order??
        list_of_words = []
        for item in data:
            for i, j in enumerate(item):
                if j not in list_of_words:
                    list_of_words.append(j)
        # print(list_of_words)
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


if __name__ == '__main__':
    pre = Preprocessor()

    plus_tweets_data = pre.file_reader("twtr_plus.txt")
    minus_tweets_data = pre.file_reader("twtr_minus.txt")
    stopwords_list = pre.file_reader("stop_words.txt")

    plus_tweets_data = pre.normalizer(plus_tweets_data)
    minus_tweets_data = pre.normalizer(minus_tweets_data)

    tokenized_plus_data = pre.tokenizer(plus_tweets_data)
    tokenized_minus_data = pre.tokenizer(minus_tweets_data)

    cleaned_plus_tweets = pre.remove_stop_words(tokenized_plus_data, stopwords_list, False)
    cleaned_minus_tweets = pre.remove_stop_words(tokenized_minus_data, stopwords_list, False)

    # Total dataset
    tweets = cleaned_plus_tweets + cleaned_minus_tweets

    ngram_plus_tweets = pre.get_ngram(cleaned_plus_tweets)
    ngram_minus_tweets = pre.get_ngram(cleaned_minus_tweets)

    # Extracted features for each class
    # plus_feature_vector = pre.feature_vector(cleaned_plus_tweets)
    # minus_feature_vector = pre.feature_vector(cleaned_minus_tweets)

    # This is our tweets tokens merged which act as the features
    # feature_vector = plus_feature_vector + minus_feature_vector
    # feature_vector = list(dict.fromkeys(feature_vector))

    # plus_vectors = []
    # for row in cleaned_plus_tweets:
    #     vec = pre.vecotrizer(feature_vector, row, 3)
    #     vec.insert(len(vec), 3)
    #     plus_vectors.append(vec)
    #
    # minus_vectors = []
    # for row in cleaned_minus_tweets:
    #     vec = pre.vecotrizer(feature_vector, row, 4)
    #     vec.insert(len(vec), 4)
    #     minus_vectors.append(vec)
    #
    # vectors = plus_vectors + minus_vectors
    # rand_inx = randrange(150, 1150)
    # test_row = vectors[rand_inx]
    # print("Test data:", test_row)
    # neighbors = pre.get_neighbors(vectors, test_row, 8)
    # predicted_class = pre.get_vote(neighbors)
    # print("Data is from class ", test_row[-1], ". ", "Predicted as class ", predicted_class, ".", sep="")
    #
    # all_docs = plus_tweets_data + minus_tweets_data
    #
    # for i, doc in enumerate(neighbors):
    #     print("همساده:", all_docs[i])
    # print("تست:", all_docs[rand_inx])







    # data_entries = pre.file_reader("Entries.txt")
    # pre.check_entries(cleaned_plus_tweets, data_entries)

    # Remove due to being redundant and poor performance
    # plus_words = pre.list_words(cleaned_plus_tweets)
    # minus_words = pre.list_words(cleaned_minus_tweets)

# https://stackoverflow.com/questions/49413139/how-do-i-make-arrays-to-be-of-the-same-length

# cProfile.run('word_util.get_ngram()') should run the function this way not plus the main call!
