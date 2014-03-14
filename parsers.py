import lxml.etree
import re

parser = lxml.etree.XMLParser(recover=True)

class Sense():
	def __init__(self, node):
		d = dict(node.items())
		self.id = d["id"]
		self.wordnet = d["wordnet"]
		self.gloss = d["gloss"]
		self.examples = d["examples"].split(" | ")

	def __str__(self):
		return self.id

	def __repr__(self):
		return str(self)

	def __iter__(self):
		return iter(self.examples)

class LexElt():
	def __init__(self, node):
		d = dict(node.items())
		(word, tag) = re.match(r"(.*)\.(.*)", d["item"]).groups()
		self.name = word
		self.tag = tag
		self.num = d["num"]
		self.senses = [Sense(s) for s in node.getchildren()]

	def __str__(self):
		return "%s.%s" % (self.word, self.tag)

	def __repr__(self):
		return str(self)

	def __iter__(self):
		return iter(self.senses)

def parse_dictionary(filename = "dictionary.xml"):
	tree = lxml.etree.parse(filename, parser)
	root = tree.getroot()
	lexelts = root.getchildren()
	return [LexElt(l) for l in lexelts]

class DataEntry():
	def __init__(self, line):
		print line
		l = re.match(r"(.+?)\.(.+?) \| (.+?) \| ?(.*?) ?%% (.+) %% ?(.*)", line).groups()
		self.word = l[0]
		self.tag = l[1]
		self.sense_id = int(l[2])
		self.prev_context = l[3].split(" ")
		self.target = l[4]
		self.next_context = l[5]

	def __str__(self):
		return "%s.%s: %d" % (self.word, self.tag, self.sense_id)

	def __repr__(self):
		return str(self)

def parse_data_file(filename):
	with open(filename, "r") as f:
		return [DataEntry(line) for line in f]

