import nltk 
import re 
from data_parser import parse_data_file
from random import shuffle 

"""
TODO: 
   - use word ranking for importance in features? 
"""
# if you want to use weighting and it just zeros out, remove zeros and roll with it 
# http://stackoverflow.com/questions/11911469/tfidf-for-search-queries
def co_occurance_window(prev, next, n=4): 
	""" Build feature of set of tokens within some n token distance of target """
	prev = nltk.word_tokenize(prev) 
	next = nltk.word_tokenize(next) 
	prev_window = prev[-n:]
	next_window = next[:n]
	return prev_window + next_window


def co_location_window(prev, next, n=4):
	feature = [] 
	prev = nltk.word_tokenize(prev) 
	next = nltk.word_tokenize(next) 
	for i in range(-n, n+1): 
		if i < 0 and abs(i) <= len(prev):  
			word = prev[i]
			feature += (word, i)
		if i > 0 and i < len(next): 
			word = next[i]
			feature += (word, i)
	return feature 


def train_model(examples, fo=0):
	"""
	Train supervised WSD model with given training data. 

		examples: text of training file containing examples
		fo: option to select what feature vector you want to use 

	"""
	word_appearances = {} 
	sense_appearances = {}
	feature_appearances = {} 
	feature_options = {0: co_occurance_window, 
						1: co_location_window
	}

	# parse examples and update dictionaries 
	for example in examples: 
		features = feature_options[0](example.prev_context, example.next_context) 

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


def smooth_model(words, senses, features): 
	""" 
	Uses absolute discounting to make room for unknown words, 
	but doesn't really work very well because it gives them too much 
	probability mass. Absolute discounting is based on the fact that 
	Good-Turing smoothing generally removes somewhere around .75 off each 
	count (based off Stanford NLP video about GTS)
	"""  
	for word in words:
		for sense in senses[word]: 
			total = 0
			for feature, count in features[(word, sense)].iteritems(): 
				total += 1 
				features[(word, sense)][feature] = count - 0.75 
			features[(word,sense)]['unk'] = total * 0.75 


def predict(example, words, senses, features, fo=0): 
	""" Predict the sense of a given example. """
	feature_options = {0: co_occurance_window, 
						1: co_location_window
	}

	probabilities = {} 
	
	if example.word not in words:
		return -1, -1

	total_occurances = words[example.word]
	local_feats = feature_options[0](example.prev_context, example.next_context)

	# go through all the senses to find highest probability 
	for sense, sense_occurances in senses[example.word].iteritems(): 
		prob = float(sense_occurances) / total_occurances # P(s_i)
		for f in local_feats: 
			if f in features[(example.word, sense)]: 
				feat_occurances = features[(example.word, sense)][f]
				prob_feat_given_sense = float(feat_occurances) / sense_occurances # P(f_i | s_i)
			else: 
				# handle unseen features 
				prob_feat_given_sense = 0.00001
			prob *= prob_feat_given_sense

		probabilities[sense] = prob

	prediction = max(probabilities.iterkeys(), key =lambda k: probabilities[k])
	confidence = probabilities[prediction]

	# extension 1: check to maybe change prediction if anyhting is within 10 percent
	for sense, prob in probabilities.iteritems(): 
		if sense != prediction and prob >= confidence * 0.1:
			prediction = tiebreaker(example, senses, features, prediction, sense)

	return prediction, confidence


def tiebreaker(example, senses, features, pred, other):
	""" 
	Check to switch prediction if other sense's confidence is within 
	10% of the predicted sense's confidence. For each feature, see if it 
	occurs over 80% of time in one of the trained senses' feature spaces. 
	Predict the sense with the most number of these big words. 
	"""
	feature_options = {0: co_occurance_window, 
						1: co_location_window
	}
	# get the features 
	feats = feature_options[0](example.prev_context, example.next_context)
	bigwords_pred = 0 
	bigwords_other = 0 

	# see if any feat occurs over 80% of time in sense's trained space 
	for f in feats:
		if f in features[(example.word, pred)]:
			feat_occurances = features[(example.word, pred)][f]
			sense_occurances = senses[example.word][pred]
			prob_feat = float(feat_occurances) / sense_occurances
			if prob_feat >= 0.80: 
				bigwords_pred += 1 

	 	if f in features[(example.word, other)]:
			feat_occurances = features[(example.word, other)][f]
			sense_occurances = senses[example.word][other]
			prob_feat = float(feat_occurances) / sense_occurances
			if prob_feat >= 0.90:
				bigwords_other += 1

	# predict the sense with highest number of important words 
	if bigwords_other > bigwords_pred: 
		return other
	else:
		return pred 


def test_model(examples, words, senses, features): 
	""" Predict each test example and return accuracy. """
	num_correct = 0.0 
	soft_score = 0.0 
	total = len(examples) 
	example_num = 0

	for example in examples:
		example_num += 1
		prediction, confidence = predict(example, words, senses, features, 0)

		if confidence == -1: 
			# test example wasn't found in trained model 
			total -= 1
		else: 
			# soft-scoring approach - TODO: do we divide??? 
			if prediction == example.sense_id:
				soft_score += confidence
			else:
				soft_score -= confidence
			
			# regular accuracy 
			num_correct += float(prediction == example.sense_id)
	
	# print "Accuracy:", float(num_correct) / total
	# print "Soft-Score:", soft_score
	return float(num_correct) / total


# test model 
train_exs = parse_data_file('training_data.data')
test_exs = parse_data_file('test_data.data')
validation_exs = parse_data_file('validation_data.data')

# n_values = [1, 5, 10, 100, 1000, 10000, 100000]    # use this for ext3 
n_values = [1000000]								 # use this for everything else 
print "n, accuracy"
for n in n_values: 
	shuffle(train_exs) # only important for ext3 
	words, senses, features = train_model(train_exs[:n], fo=0)
	accuracy = test_model(validation_exs, words, senses, features)
	# accuracy = test_model(test_exs, words, senses, features)
	print str(n) + "," + str(accuracy)

