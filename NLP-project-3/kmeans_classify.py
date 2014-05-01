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
		doc = preprocess(sentence.text)
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

km = KMeans(n_clusters = 8)
train_feats = map(feature_extractor, docs)
print "training features"
km.fit(train_feats)
print "km fit"

def classify(sentence):
	feature = feature_extractor(preprocess(sentence.text))
	prediction = km.predict(feature)[0]
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
hmm = trainer.train_supervised(labelled_sequences)
print "hmm"

# sents = ['pos', 'neu', 'neg']
# confusion_hmm = {(i,j) : 0 for i in sents for j in sents}
# total = 0
# for review in validation_reviews:
# 	print "========================"
# 	print review.header.strip()
# 	actual = [s.sentiment for s in review]
# 	guess_kmc = map(classify, review)

# 	print "actual"
# 	print [sent_map[a] for a in actual]
# 	print "kmc guess"
# 	print guess_kmc
# 	guess_hmm = hmm.best_path(guess_kmc)
# 	print "hmm guess"
# 	print [sent_map[a] for a in guess_hmm]

# 	for i in range(len(review)):
# 		total += 1
# 		a = actual[i]
# 		h = guess_hmm[i]
# 		confusion_hmm[(a,h)] += 1
# 	print "========================"

# print "hmm confusion"
# print confusion_hmm

# correct_hmm = sum(confusion_hmm[(i,i)] for i in sents)
# prec_hmm = 100.0 * (float(correct_hmm) / total)
# print "hmm: %d" % prec_hmm

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


