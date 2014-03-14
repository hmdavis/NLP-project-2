"""
Installing lxml is getting fugged up so this is a temp
"""
import re 

class DataEntry():
	def __init__(self, line):
		# pro regex
		l = re.match(r"(.+?)\.(.+?) \| (.+?) \| ?(.*?) ?%% (.+) %% ?(.*)", line).groups()
		self.word = l[0]
		self.tag = l[1]
		self.sense_id = int(l[2])
		self.prev_context = l[3]
		self.target = l[4]
		self.next_context = l[5]

	def __str__(self):
		return "%s.%s: %d" % (self.word, self.tag, self.sense_id)

	def __repr__(self):
		return str(self)

def parse_data_file(filename):
	with open(filename, "r") as f:
		return [DataEntry(line) for line in f]