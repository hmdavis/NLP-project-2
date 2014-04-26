import re
from collections import defaultdict
class ReviewSentence():
	def __init__(self, line):
		self.sentiment = line[:3]
		self.text = line[4:]


class Review():
	def __init__(self, header):
		self.header = header
		a, b, c = header.split("_")
		self.review_category = a
		self.review_sentiment = b
		self.sentences = []

	def add_sentence(self, line):
		s = ReviewSentence(line)
		self.sentences.append(s)

	def __iter__(self):
		return iter(self.sentences)

	def __getitem__(self, index):
		return self.sentences[index]

def parse_review_file(filename = "training_data.txt"):
	reviews = []
	curr_review = None
	with open(filename, "r") as f:
		for line in f:
			if curr_review:
				if line == '\n':
					reviews.append(curr_review)
					curr_review = None
				else:
					curr_review.add_sentence(line)
			else:
				if line != '\n':
					curr_review = Review(line)
		if curr_review:
			reviews.append(curr_review)
	return reviews


# reviews = parse_review_file()
# r = reviews[0]

def get_pos_neg_words():
	pos = set()
	neg = set()
	with open('opinion-lexicon-English/positive-words.txt', 'r') as f:
		for line in f:
			if line[0] != ';':
				pos.add(line.strip())
	with open('opinion-lexicon-English/negative-words.txt', 'r') as f:
		for line in f:
			if line[0] != ';':
				neg.add(line.strip())
	return pos, neg

clue_re = re.compile(r"type=(.*) len=1 word1=(.*) pos1=(.*) stemmed1=(.*) priorpolarity=(.*)")
class ClueDict:
	def __init__(filename):
		self.clues = defaultdict(lambda : [])
		with open(filename, 'r') as f:
			for line in f:
				s = line.strip()
				clue = Clue(s)
				self.clues[clue.word].append(clue)
	def __contains__(self, word):
		return word in self.clues
	def __getitem__(self, word):
		return self.clues[word]


class Clue:
	def __init__(line):
		match = clue_re.match(line)
		groups = match.groups()
		self.strong = (groups[0] == 'strongsubj')
		self.word = groups[1]
		self.pos = groups[2]
		self.stemmed = (groups[3] == 'y')
		self.polarity = groups[4][:3]
	def __str__(self):
		return "word: %s, polarity: %s" % (self.word, self.polarity)
	def __repr__(self):
		return str(self)


