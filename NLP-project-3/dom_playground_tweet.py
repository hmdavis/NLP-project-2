from parsers import *
import nltk
from nltk.corpus import stopwords
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.stem.lancaster import LancasterStemmer

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
	# st = LancasterStemmer()
	ret = [w.lower() for w in parse_words_only(sentence)]
	return ret

review_list = parse_review_file("training_data.txt")

num_reviews = len(review_list)
num_training = 3 * num_reviews / 4
num_validation = num_reviews - num_training

training_reviews = review_list[:num_training]
validation_reviews = review_list[num_training:]

sents = {'pos' : [], 'neg' : [], 'neu' : []}

for review in training_reviews:
	for sentence in review:
		sent = sentence.sentiment
		sents[sent].append(sentence.text)


neglist = sents['pos']
poslist = sents['neg']
neulist = sents['neu']

#Creates a list of tuples, with sentiment tagged.
postagged = [(sentence, 'pos') for sentence in poslist]
negtagged = [(sentence, 'neg') for sentence in neglist]
neutagged = [(sentence, 'neu') for sentence in neulist]
 
#Combines all of the tagged tweets to one large list.
taggedtweets = postagged + negtagged + neutagged

print "tagged"


tweets = []
 
#Create a list of words in the tweet, within a tuple.
for (sentence, sentiment) in taggedtweets:
	word_filter = preprocess(sentence)
	tweets.append((word_filter, sentiment))
 
#Pull out all of the words in a list of tagged tweets, formatted in tuples.
def getwords(tweets):
	allwords = []
	for (words, sentiment) in tweets:
		allwords.extend(words)
	return allwords
 
#Order a list of tweets by their frequency.
def getwordfeatures(listoftweets):
#Print out wordfreq if you want to have a look at the individual counts of words.
	wordfreq = nltk.FreqDist(listoftweets)
	words = wordfreq.keys()
	return words
 
#Calls above functions - gives us list of the words in the tweets, ordered by freq.
# print getwordfeatures(getwords(tweets))

wordlist = set(getwords(tweets))
wordlist = set([i for i in wordlist if not i in stopwords.words('english')])
ind_to_word = list(wordlist)
word_to_ind = dict((j, i) for (i, j) in enumerate(ind_to_word))
num_words = len(ind_to_word)

print "wordlist"

def feature_extractor(doc):
	docwords = set(doc)
	vector = {i : (ind_to_word[i] in docwords) for i in xrange(num_words)}
	return vector
 
#Creates a training set - classifier learns distribution of true/falses in the input.
training_set = nltk.classify.apply_features(feature_extractor, tweets)
print "training set"
classifier = nltk.NaiveBayesClassifier.train(training_set)
print "classifier"

# print classifier.most_informative_features()

def classify(sentence):
	return classifier.classify(feature_extractor(preprocess(sentence.text)))
	
def label_review(review):
	labelled_sequence = []
	for sentence in review:
		symb = classify(sentence)
		sentiment = sentence.sentiment
		labelled_sequence.append((sentiment, symb))
	return labelled_sequence

labelled_seqeuences = map(label_review, training_reviews)

trainer = HiddenMarkovModelTrainer()
hmm = trainer.train_supervised(labelled_seqeuences)
print "hmm"

correct_bayes = correct_hmm = total = 0

for review in validation_reviews:
	print "========================"
	print review.header.strip()
	actual = [s.sentiment for s in review]
	guess_bayes = map(classify, review)

	print "actual"
	print actual
	print ""
	print "bayes"
	print guess_bayes
	print ""
	guess_hmm = hmm.best_path(guess_bayes)
	print "hmm"
	print guess_hmm

	for i in range(len(review)):
		total += 1
		if actual[i] == guess_bayes[i]:
			correct_bayes += 1
		if actual[i] == guess_hmm[i]:
			correct_hmm += 1

	print "========================"

prec_bayes = 100.0 * (float(correct_bayes) / total)
prec_hmm = 100.0 * (float(correct_hmm) / total)
print "bayes: %d, hmm: %d" % (prec_bayes, prec_hmm)
