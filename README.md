# Word Attraction Model for Unsupervised Key Word Extraction

This is an unofficial implementation of __keyword__ __extration__ method proposed in this paper:

[Wang, R., Liu, W., & McDonald, C. (2014, July). Corpus-independent generic keyphrase extraction using word embedding vectors. In Software Engineering Research Conference (Vol. 39).](https://pdfs.semanticscholar.org/bd37/94c777af5ba363abae5708050ea78ecc97e2.pdf)


Briefly, the method is a variantion of textrank but take semantic similarity of words into consideration.
The pairwise semantic similarity is measured by cosine distance of word vectors of two words.


## Prerequisite:

You should have a pretrained word2vec model before utilizing the method to extract keywords. 
There's a demo model, which I trained with very few data and of very low embedding dimension, 
avaible at __/data/__. Customized embedding model should work just fine if being trained via 
[https://github.com/dav/word2vec](https://github.com/dav/word2vec) and setting the -binary to 0.
Or equivalently, use the repo link __@word2vec__.

Beside, NLTK is used for stemming words but not required. 
It should be a good practice to stem word in english, but it can be trivial if the target language doesn't change form of words. If NLTK is not installed, it will automatically skip stemming.

## Run

```python
from word2vecc_loader import Word2Vec
from word_attraction import WordAttraction

w2v = Word2Vec("data/text8-vector.bin")
wa = WordAttraction(w2v)
print(wa.extract_main(text, max_words=3, stem=False))
```

Try play with __demo.py__. 

## Further

English stopwords are hard coded in stopwords.py, 
if working on other language, it's better to modify this list accordingly. 
Not doing so won't lead to software problems but the keywords might be filled up with those very frequent but less informative words. 
