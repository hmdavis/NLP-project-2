import lxml.etree
import re
# regular xml parser has no recovery option
parser = lxml.etree.XMLParser(recover=True)

class Sense():
	def __init__(self, node, word):
		d = dict(node.items())
		self.word = word
		self.id = int(d["id"])
		self.wordnet = d["wordnet"]
		self.gloss = d["gloss"]
		self.examples = d["examples"].split(" | ")

	# easier debugging
	def __str__(self):
		return "%s: %d" % (self.word, self.id)

	def __repr__(self):
		return str(self)

	# overload as list of examples (call "sense[1]")
	def __len__(self):
		return len(self.examples)

	def __getitem__(self, key):
		return self.examples[key]

	def __iter__(self):
		return iter(self.examples)

class LexElt():
	def __init__(self, node):
		d = dict(node.items())
		(word, tag) = re.match(r"(.*)\.(.*)", d["item"]).groups()
		self.word = word
		self.tag = tag
		self.num = d["num"]
		word = "%s.%s" % (self.word, self.tag)
		self.senses = [Sense(s, word) for s in node.getchildren()]

	# easier debugging
	def __str__(self):
		return "%s.%s" % (self.word, self.tag)

	def __repr__(self):
		return str(self)

	# overload as list of senses (call "lexelt[1]")
	def __len__(self):
		return len(self.senses)

	def __getitem__(self, key):
		return self.senses[key]

	def __iter__(self):
		return iter(self.senses)

def parse_dictionary(filename = "dictionary.xml"):
	tree = lxml.etree.parse(filename, parser)
	root = tree.getroot()
	lexelts = root.getchildren()
	return [LexElt(l) for l in lexelts]

class DataEntry():
	def __init__(self, line):
		# pro regex
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

print parse_dictionary()[-1][-1]