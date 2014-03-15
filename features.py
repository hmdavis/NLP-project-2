import nltk 
import re 
from data_parser import parse_data_file


# if you want to use weighting and it just zeros out, remove zeros and roll with it 
# http://stackoverflow.com/questions/11911469/tfidf-for-search-queries
def co_occurence_window(prev, next, n=4): 
	''' Build feature of set of tokens within some n token distance of target ''' 
	prev = nltk.word_tokenize(prev) 
	next = nltk.word_tokenize(next) 
	# TODO: use tdif weighting to only use important words! 
	prev_window = prev[-n:]
	next_window = next[:n]
	return prev_window + next_window


def co_location_window(prev, next, n=4):
	pass


def ngrams(prev, next, n=4): 
	pass 


def pos_tags(prev, next, n=4): 
	pass 


def train_model(examples, fo=0):
	'''
	Train supervised WSD model with given training data. 

		examples: text of training file containing examples
		fo: option to select what feature vector you want to use 

	'''
	word_appearances = {} 
	sense_appearances = {} 
	feature_appearances = {} 
	# feature_options = {0: co_occurence_window, 
	# 					1: co_location_window, 
	# 					2: ngrams, 
	# 					3: pos_tags
	# }

	# just for testing 
	feature_options = {0: co_occurence_window}

	# parse examples and update dictionaries 
	for example in examples: 

		features = feature_options[fo](example.prev_context, example.next_context, 4) 

		# update the count of target word appearances
		if example.word in word_appearances: 
			word_appearances[example.word] += 1 
		else: 
			word_appearances[example.word] = 1 
			sense_appearances[example.word] = {} 

		# update the count of sense appearances 
		if example.sense_id in sense_appearances[example.word].keys():
			sense_appearances[example.word][example.sense_id] += 1 
		else: 
			sense_appearances[example.word][example.sense_id] = 1

		# update the feature counts for specific senses 
		for f in features: 
			if (example.word, example.sense_id) not in feature_appearances: 
				feature_appearances[(example.word, example.sense_id)] = {} 		
			if f in feature_appearances[(example.word, example.sense_id)].keys():
				feature_appearances[(example.word, example.sense_id)][f] += 1
			else: 
				feature_appearances[(example.word, example.sense_id)][f] = 1 

	return word_appearances, sense_appearances, feature_appearances


def predict(example, words, senses, features): 
	''' Predict the sense of a given example. '''
	probabilities = {} 
	total_occurances = words[example.word]

	# go through all the senses 
	for sense, sense_occurances in senses[example.word].iteritems(): 
		prob = float(sense_occurances) / total_occurances # P(s_i)

		for f, feat_occurances in features[(example.word, sense)].iteritems(): 
			prob_feat_given_sense = float(feat_occurances) / sense_occurances # P(f_i | s_i)
			prob *= prob_feat_given_sense

		probabilities[sense] = prob 

	# pretty sure this returns the key with the maximum value 
	prediction = max(probabilities.iterkeys(), key =lambda k: probabilities[k])

	return prediction


# just make sure it runs nicely 
test_exs = parse_data_file('test_data.data')
train_exs = parse_data_file('training_data.data')
words, senses, features = train_model(train_exs, fo=0)
predict(train_exs[0], words, senses, features)

