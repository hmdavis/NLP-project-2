from parsers import *
import nltk
from nltk.corpus import stopwords
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.tag.crf import MalletCRF
from nltk.stem.lancaster import LancasterStemmer
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import linear_model
from operator import or_
from collections import defaultdict

word_only_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w[\w\'-]*\w?|\?|\$")
def parse_words_only(s):
	l = word_only_tokenizer.tokenize(s)
	return l
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
def lemmatize(s):
	return lmtzr.lemmatize(s)
st = LancasterStemmer()
def stem(s):
	return st.stem(s)
def preprocess(sentence):
	ret = [lemmatize(w.lower()) for w in parse_words_only(sentence)]
	return ret

review_list = parse_review_file("training_data.txt")

num_reviews = len(review_list)
num_training = num_reviews
num_validation = num_reviews - num_training

training_reviews = review_list[:num_training]
validation_reviews = review_list[num_training:]

sent_map = {"pos" : 1, "neu" : 0, "neg" : -1}
map_sent = dict((j,i) for (i,j) in sent_map.items())

docs = []
tagged_docs = []

for review in training_reviews:
	for sentence in review:
		sent = sentence.sentiment
		text = preprocess(sentence.text)
		bigrams = []
		for i in xrange(len(text) - 1):
			bigrams.append((text[i], text[i + 1]))
		doc = text + bigrams
		docs.append(set(doc))
		tagged_docs.append((doc, sent))

vocab = set(reduce(or_, docs))
vocab = set([i for i in vocab if not i in stopwords.words('english')])

thresh = 2
counts = {i : 0 for i in vocab}
for i in vocab:
	for doc in docs:
		if i in doc:
			counts[i] += 1
			if counts[i] > thresh:
				break

vocab = set(i for i in vocab if counts[i] > thresh)
print len(vocab)
print "wordlist"

def normalize(v):
	l = sum(i * i for i in v)
	if l == 0: return v
	r = tuple(i / float(l) for i in v)
	return r

def feature_extractor(doc):
	docwords = set(doc)
	vector = tuple(float(i in docwords) for i in vocab)
	normalized = normalize(vector)
	return normalized

X = []
y = []
for (sentence, sentiment) in tagged_docs:
	feature = feature_extractor(sentence)
	X.append(feature)
	y.append(sent_map[sentiment])

print "features"

clf = svm.LinearSVC()
clf.fit(X, y)

print "svm"

def classify(sentence):
	feature = feature_extractor(preprocess(sentence.text))
	prediction = clf.predict(feature)[0]
	# probs = clf.predict_proba(feature)[0]
	# if probs[prediction + 1] > 0.5:
	# 	return prediction
	# else:
	# 	return 0
	return prediction
	
def label_review(review):
	labelled_sequence = []
	for sentence in review:
		symb = classify(sentence)
		sentiment = sentence.sentiment
		labelled_sequence.append((symb, sentiment))
	return labelled_sequence


labelled_sequences = map(label_review, training_reviews)
print "labels"
trainer = HiddenMarkovModelTrainer()
hmm = trainer.train_supervised(labelled_sequences, estimator = nltk.LidstoneProbDist)
print "hmm"

# sents = ['pos', 'neu', 'neg']
# confusion_svm = {(i,j) : 0 for i in sents for j in sents}
# confusion_hmm = {(i,j) : 0 for i in sents for j in sents}
# total = 0
# for review in validation_reviews:
# 	print "========================"
# 	print review.header.strip()
# 	actual = [s.sentiment for s in review]
# 	guess_svm = map(classify, review)

# 	print "actual"
# 	print [sent_map[a] for a in actual]
# 	print "svm guess"
# 	print guess_svm
# 	guess_hmm = hmm.best_path(guess_svm)
# 	print "hmm guess"
# 	print [sent_map[a] for a in guess_hmm]

# 	for i in range(len(review)):
# 		total += 1
# 		a = actual[i]
# 		b = map_sent[guess_svm[i]]
# 		h = guess_hmm[i]
# 		confusion_svm[(a,b)] += 1
# 		confusion_hmm[(a,h)] += 1
# 	print "========================"

# print "svm confusion"
# print confusion_svm
# print "hmm confusion"
# print confusion_hmm

# correct_svm = sum(confusion_svm[(i,i)] for i in sents)
# correct_hmm = sum(confusion_hmm[(i,i)] for i in sents)
# prec_svm = 100.0 * (float(correct_svm) / total)
# prec_hmm = 100.0 * (float(correct_hmm) / total)
# print "svm: %d, hmm: %d" % (prec_svm, prec_hmm)

test_reviews = parse_review_file("test_data_no_true_labels.txt")
print "Id,Answer"
id = -1
for review in test_reviews:
	actual = [s.sentiment for s in review]
	guess_bayes = map(classify, review)
	guess_hmm = hmm.best_path(guess_bayes)
	for h in guess_hmm:
		id += 1
		print "%d,%d" % (id, sent_map[h])


