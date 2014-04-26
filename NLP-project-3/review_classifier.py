from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.model.ngram import NgramModel
from nltk import wordpunct_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
from parsers import *
from operator import itemgetter
from collections import defaultdict
from math import sqrt

word_only_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w[\w\'-]*\w?")
def parse_words_only(s):
    """
    Returns a list of words with no punctuation.
    """
    l = word_only_tokenizer.tokenize(s)
    return l
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
def lemmatize(s, pos = None):
    """
    Returns the lemma os s
    Optional POS tag
    """
    if pos:
        return lmzr.lemmatize(s, pos)
    return lmtzr.lemmatize(s)

def preprocess(sentence):
    """
    converts a sentence into a list of lowercse, lemmatized words
    """
    st = LancasterStemmer()
    ret = [w.lower() for w in parse_words_only(sentence)]
    return ret

training_reviews = parse_review_file("training_data.txt")

clue_dict = ClueDict("subjectivity_clues/clues.tff")

def classify(sentence):
    score = 0
    for word in sentence:
        if word in clue_dict:
            clues = clue_dict[word]
            clue = clues[0]
    
def label_review(review):
    labelled_sequence = []
    for sentence in review:
        symb = classify(preprocess(sentence.text))
        sentiment = sentence.sentiment
        labelled_sequence.append((sentiment, symb))
    return labelled_sequence

labelled_seqeuences = map(label_review, training_reviews)

trainer = HiddenMarkovModelTrainer()
hmm = trainer.train_supervised(labelled_seqeuences)

def tag(review):
    print review.header
    sentences = [preprocess(s.text) for s in review]
    symbs = map(classify, sentences)
    path = hmm.best_path(symbs)
    print [s.sentiment for s in review]
    print map(classify, sentences)
    print path

validation_reviews = parse_review_file("validation_data.txt")
for review in validation_reviews[:10]:
    tag(review)


