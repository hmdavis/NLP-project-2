import networkx as nx
from itertools import *
from collections import defaultdict

def random_graph(num_students):
	return nx.fast_gnp_random_graph(num_students, 0.5)

def optimal_FRP(G, l):
	assert(sum(l) >= len(G.nodes()))
	cliques_by_size = defaultdict(lambda : [])
	max_room_size = max(l)
	cliques_by_size[1] = [(i,) for i in G.nodes()]
	prev_clique_size = 1
	while prev_clique_size < max_room_size:
		for n in G.nodes():
			for clique in cliques_by_size[prev_clique_size]:
				if n in clique:
					continue
				is_clique = True
				for m in clique:
					if not G.has_edge(n, m):
						is_clique = False
						break
				if is_clique:
					new_clique = (n,) + clique
					cliques_by_size[prev_clique_size + 1].append(clique)
		prev_clique_size += 1
	

