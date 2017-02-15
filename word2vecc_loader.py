import math
import re


class Word2Vec(object):
    """ the class is the interface to load word embedding model from Google Word2Vec code, from following link:
        https://code.google.com/archive/p/word2vec/
        Tt loads txt format word2vec model file instead of binary file.
        (The output format can be set as "-binary 0" when running Word2Vec)
        This class also provides query and searching method."""
    def __init__(self, filename):
        self.__MAX_DISTANCE = 100000
        self.__word_embedding = {}
        with open(filename, 'r') as fp:
            # skip first line: vocabulary size, embedding dimension
            info = fp.readline()
            self.__dimension = int(info.split()[1])
            for line in fp:
                line_split = line.strip().split()
                self.__word_embedding[line_split[0]] = [float(token) for token in line_split[1:]]

    def get_vector(self, word):
        """word can be a single word or white-space delimited phrase,
        return zeroes vector when word not in dictionary"""
        word = word.lower()
        words = re.split(r"[\s\t\n\-]+", word)
        vector = [0] * self.__dimension
        for w in words:
            if w in self.__word_embedding:
                for i in range(self.__dimension):
                    vector[i] += self.__word_embedding[w][i]
        return vector

    def distance(self, word1, word2):
        """Returns cosine distance of word1 and word2, any of which can be single word or whitespace delimited phrase"""
        vec1, vec2 = self.get_vector(word1), self.get_vector(word2)
        length1 = sum(i ** 2 for i in vec1)
        length2 = sum(i ** 2 for i in vec2)
        product = sum(vec1[i] * vec2[i] for i in range(self.__dimension))
        if length1 == 0 or length2 == 0:
            return 0
        return product / math.sqrt(length1 * length2)

    def __contains__(self, item):
        return item in self.__word_embedding

    def __getitem__(self, item):
        return self.__word_embedding[item]
