from parsers import parse_dictionary, parse_data_file, DataEntry
import nltk
from nltk.stem.lancaster import LancasterStemmer
from math import log
from collections import defaultdict, Counter

# ' handles contractions and the like
# - handles compound words
word_only_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w[\w\'-]*\w?")
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

def preprocess(sentence):
	"""
	converts a sentence into a list of lowercse, lemmatized words
	"""
	st = LancasterStemmer()
	return [lemmatize(st.stem(w.lower())) for w in parse_words_only(sentence)]

class Dictionary():
	"""
	implements a dictionary (in the language sense)
	"""
	def __init__(self, lexelts):
		# create lexelts table
		self.lexelts = {str(lex) : lex for lex in lexelts}
		# compute idfs
		self.doc_counts = defaultdict(lambda : 0)
		self.doc_num = 0
		self.compute_idfs()

	def traverse_dict_docs(self):
		"""
		Yield glosses and examples as documents
		"""
		for lex in self.lexelts.itervalues():
			for sense in lex:
				yield sense.gloss
				for ex in sense.examples:
					yield ex

	def compute_idfs(self):
		for doc in self.traverse_dict_docs():
			self.doc_num += 1
			# only consider each word once
			for word in set(doc):
				self.doc_counts[word] += 1

	def idf(self, word):
		"""
		Returns the inverse document frequency of t
		adds 1 to divisor to stop DB0 errors
		"""
		return log(self.doc_num / (self.doc_counts[word] + 1.0))

	def __getitem__(self, key):
		"""
		look up a word in the dictionary
		call by Dictionary()[word]
		checks word, word.n, word.v, and lemma(word)
		"""
		for k in [key, key + ".n", key + ".v", lemmatize(key)]:
			if k in self.lexelts:
				return self.lexelts[k]
		raise KeyError(key)

	def __contains__(self, key):
		for k in [key, key + ".n", key + ".v", lemmatize(key)]:
			if k in self.lexelts:
				return True
		return False

# load dictionary
lexelts = parse_dictionary("dictionary.xml")

# preprocess glosses and examples
for lex in lexelts:
	for sense in lex:
		sense.gloss = preprocess(sense.gloss)
		sense.examples = map(preprocess, sense.examples)


dictionary = Dictionary(lexelts)

idf_thresh = 3
stopwords = set(nltk.corpus.stopwords.words('english'))
def key_words(l):
	"""
	input: l - a list of words
	output: a list of content words
	uses idf and stopwords
	"""
	out = []
	for i in l:
		if i not in stopwords:
			if dictionary.idf(i) > idf_thresh:
				out.append(i)
	return out

def get_context(sentence):
	"""
	input: sentence - text
	output list of words deemed important also in dictionary
	"""
	pruned = preprocess(sentence)
	# print pruned
	out = []
	for word in key_words(pruned):
		if word in dictionary:
			out.append(word)
	return out

def get_signature(sense):
	"""
	returns the set of all words in a sense
	"""
	sig = set(sense.gloss)
	for ex in sense:
		sig |= set(ex)
	return set(key_words(sig))

def get_overlap(sense, context):
	"""
	determine the fit of a sense to a context
	"""
	sig = get_signature(sense)
	ct = set(context)
	for word in context:
		lex = dictionary[word]
		for s in lex:
			ct |= set(s.gloss)
			for ex in s:
				ct |= set(ex)
	ct = set(key_words(ct))
	return len(ct & sig)

def lesk(lex, sentence):
	"""
	implements the lesk algorithm for choosing word senses
	"""
	best_sense = 1
	max_overlap = 0
	# choose relavant words
	context = get_context(sentence)
	for sense in lex:
		overlap = get_overlap(sense, context)
		# print "sense: %d, overlap: %d" % (sense.id, overlap)
		if overlap > max_overlap:
			max_overlap = overlap
			best_sense = sense.id
	return best_sense

def choose_sense(de):
	"""
	input: de - DataEntry
	output: the sense id of the target word
	"""
	# choose target
	target = de.word + "." + de.tag
	# print "target:"
	# print target
	# create sentence
	sentence = "%s %s %s" % (de.prev_context, de.target, de.next_context)

	# look up lexelt
	lex = dictionary[target]

	# run lesk
	return lesk(lex, sentence)

def validate(filename, max_iter = -1):
	with open(filename, "r") as f:
		correct = total = 0
		iter_count = 0
		for line in f:
			if iter_count == max_iter: break
			de = DataEntry(line)
			total += 1
			r = choose_sense(de)
			print iter_count + 1, r, de.sense_id, "*" if r == de.sense_id else ""
			if r == de.sense_id:
				correct += 1
			iter_count += 1
		return (correct, total)

(correct, total) = validate("validation_data.data")
print "%d correct out of %d: %3.2f percent" % (correct, total, 100.0*correct / total)