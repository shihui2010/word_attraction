from stopwords import stopwords_en
import warnings
import re
try:
    from nltk.stem.porter import PorterStemmer
    STEM = True
except ImportError:
    print("warning: stem function is off")
    STEM = False


class WordAttraction(object):
    """
    Implementation of methods proposed in:
    Corpus-independent Generic Keyphrase Extraction Using Word Embedding Vectors
    POS tag filtering is excluded from process as no POS tagging tool can be applied to all languages,
    and it's time-consuming.
    """
    def __init__(self, embedding_model):
        self.__accuracy, self.__running, self.__text_length = set(), set(), set()
        self.__word_embedding = embedding_model
        self.__MAX_DISTANCE = 100000000
        self.__unigram = None
        self.__bigram = None
        self.__graph = None
        if STEM:
            self.__wnl = PorterStemmer()

    def __prepocessing(self, text, stem):
        if text is None:
            warnings.warn("Null Input")
            return
        text = re.sub(r"[:;,.\?\[\]'\!\n]", " ", text.lower())
        tokens = re.findall('\w+', text)
        self.__text_length.add(len(tokens))
        self.__unigram, self.__bigram = dict(), dict()
        last = None
        for token in tokens:
            if token in stopwords_en:
                continue
            if STEM and stem:
                token = self.__wnl.stem(token)
            self.__unigram[token] = self.__unigram.setdefault(token, 0) + 1
            if last is not None:
                last[token] = last.setdefault(token, 0) + 1
            last = self.__bigram.setdefault(token, {})

    def __distance(self, w1, w2):
        """
            returns squared Euclidean distance between embedded vectors of w1 and w2
            if any of the word can not be found in Word2Vec model, MAX_DISTANCE returned
        """
        if not (w1 in self.__word_embedding and w2 in self.__word_embedding):
            return self.__MAX_DISTANCE
        if w1 == w2:
            return 0.0
        dis = 0.0
        for i in range(len(self.__word_embedding[w1])):
            dis += (self.__word_embedding[w1][i] - self.__word_embedding[w2][i]) ** 2
        return dis

    def __attraction_score(self, w1, w2):
        """
            weight of undirected edge of w1-w2 in the graph
        """
        if w1 == w2:
            return 0.0
        freq1, freq2 = self.__unigram[w1], self.__unigram[w2]
        coor = self.__bigram[w1].setdefault(w2, 0) + self.__bigram[w2].setdefault(w1, 0)
        if coor == 0:
            return 0.0
        distance = self.__distance(w1, w2)
        return (2 * freq1 * freq2 * coor) / (distance * (freq1 + freq2))

    def __construct_graph(self):
        """setup self.__graph(undirected) based on self.__bigram(directed)"""
        self.__graph, visited = dict(), set()
        for w1 in self.__bigram:
            self.__graph.setdefault(w1, dict())
            visited.add(w1)
            for w2 in self.__bigram[w1]:
                if w2 in visited:
                    continue
                self.__graph[w1][w2] = self.__attraction_score(w1, w2)
                self.__graph.setdefault(w2, dict())[w1] = self.__graph[w1][w2]
        for w1 in self.__graph:
            wsum = sum(self.__graph[w1][token] for token in self.__graph[w1])
            if wsum == 0:
                continue
            for w2 in self.__graph[w1]:
                self.__graph[w1][w2] /= wsum

    def __scoring(self, damping=0.85, max_iter=50, converge_threshold=0.0001):
        """Calculating scores of vertices in the graph"""
        last_states = dict()
        for key in self.__graph:
            last_states[key] = 1.0 / len(self.__graph)
        iter_n = 0
        while iter_n < max_iter:
            states = dict()
            for w in last_states:
                states[w] = (1 - damping)/len(last_states)
            for w1 in last_states:
                for w2 in self.__graph[w1]:
                    states[w2] += damping * last_states[w1] * self.__graph[w1][w2]
            should_continue = False
            for w in last_states:
                diff = abs(last_states[w] - states[w]) / last_states[w]
                if diff > converge_threshold:
                    should_continue = True
                    break
            if not should_continue:
                break
            last_states = states
        return last_states

    def extract_main(self, text, output_score=True, stem=True,
                     max_words=2, damping=0.85, max_iter=50, converge_threshold=0.01):
        """
        main method for extracting keywords
        :param text: content from which keywords are extracted
        :param output_score: if True, return sorted list of tuples: (key, score). if False, return set of top keywords
        :param stem: if True, keywords returned are stemmed by PorterStemmer in nltk
        :param max_words: max number of keywords expected from text
        :param damping: damping factor for scoring
        :param max_iter: max number of iteration for scoring
        :param converge_threshold: converge condition for scoring
        """
        self.__prepocessing(text, stem=stem)
        self.__construct_graph()
        scores = self.__scoring(damping, max_iter, converge_threshold)

        tmp = sorted([(key, scores[key]) for key in scores], key=lambda x: -x[1])[:max_words]
        if output_score:
            return tmp
        return set([item[0] for item in tmp])
