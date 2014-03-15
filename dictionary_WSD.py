from parsers import parse_dictionary, parse_data_file
import nltk
from math import log
from collections import defaultdict

# load dictionary
lexelts = parse_dictionary("dictionary.xml")
dictionary = {lex.word : lex for lex in lexelts}

# ' handles contractions and the like
# - handles compound words
word_only_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w[\w\'-]*")
def parse_words_only(s):
	"""
	Returns a list of words with no punctuation.
	"""
	return word_only_tokenizer.tokenize(s)

lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
def lemmatize(s, pos = None):
	"""
	Returns the lemma os s
	Optional POS tag
	"""
	if pos:
		return lmzr.lemmatize(s, pos)
	return lmtzr.lemmatize(s)

def prune(s):
	"""
	Returns a list of lemmatized words
	"""
	return [lemmatize(w) for w in parse_words_only(s)]

# prune glosses and examples
for (word, lex) in dictionary.iteritems():
	for sense in lex:
		sense.gloss = prune(sense.gloss)
		sense.examples = map(prune, sense.examples)


# idf score (t) = num_docs / (num_docs containing t)
# glosses and examples are considered docuements
def traverse_dict_docs():
	"""
	Yield glosses and examples as documents
	"""
	for lex in lexelts:
		for sense in lex:
			yield sense.gloss
			for ex in sense.examples:
				yield ex

doc_counts = defaultdict(lambda : 0)
doc_num = 0
for doc in traverse_dict_docs():
	doc_num += 1
	# only consider each word once
	for word in set(doc):
		doc_counts[word] += 1

def idf(t):
	"""
	Returns the inverse document frequency of t
	adds 1 to divisor to stop DB0 errors
	"""
	return log(doc_num / (doc_counts[t] + 1.0))

