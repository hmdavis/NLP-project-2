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


reviews = parse_review_file()
r = reviews[0]
