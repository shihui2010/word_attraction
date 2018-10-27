from word2vecc_loader import Word2Vec
from word_attraction import WordAttraction


text = '''Keyphrase extraction from a given document is a difficult
task that requires not only local statistical information but
also extensive background knowledge. In this paper, we
propose a graph-based ranking approach that uses information
supplied by word embedding vectors as the background
knowledge. We first introduce a weighting scheme that computes
informativeness and phraseness scores of words using
the information supplied by both word embedding vectors
and local statistics. Keyphrase extraction is performed by
constructing a weighted undirected graph for a document,
where nodes represent words and edges are co-occurrence
relations of two words within a defined window size. The
weights of edges are computed by the afore-mentioned weighting
scheme, and a weighted PageRank algorithm is used to
compute final scores of words. Keyphrases are formed in
post-processing stage using heuristics. Our work is evaluated
on various publicly available datasets with documents
of varying length. We show that evaluation results are comparable
to the state-of-the-art algorithms, which are often
typically tuned to a specific corpus to achieve the claimed
results.'''

w2v = Word2Vec("data/text8-vector.bin")
wa = WordAttraction(w2v)
print(wa.extract_main(text, max_words=3, stem=False))
