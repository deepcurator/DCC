import glob, sys, time, os
import json
from rdflib import Graph, BNode, RDFS, RDF, URIRef, Literal
import rules
from matplotlib import colors as mcolors
from pyvis.network import Network
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path

cnames = ['blue', 'green', 'red', 'cyan', 'orange', 'black', 'purple', 'purple', 'purple']
sizes = [i*2 for i in [10,7,6,5,4,3,2,1,1]]

class Json2RDFParser:

	def __init__(self):
		self.RDF_graph = Graph()
		self.simplified_RDF_graph = Graph()

		self.root = BNode("Root")
		self.has_input = URIRef("http://example.org/has_input")
		self.has_op = URIRef("http://example.org/has_op")
		self.on_device = URIRef("http://example.org/on_device")

		# self.attrs = [self.root]

		# for making simplified RDF graph.
		self.interested_words = ['conv','pool','bias','save','dense','flatten','metrics','training', 'tfoptimizer', 'loss', 'adam']
		self.not_interested_words = ['varisinitialized', 'isvariableinitialized', 'init', 'save', 'gradient']

		# for binding the type
		# self.datatype_manager = DataTypeManager()

		# defining rules for parsing attrs.
		self.rules = []

		for name in dir(rules):
			value = getattr(rules, name)
			if isinstance(value, type) and issubclass(value, rules.GGenerator):
				o = value()
				if type(o).__name__ is not "GGenerator":
					self.rules.append(o)

		self.attrs = [rule.attr_uriref for rule in self.rules]

	def parse(self, jsonGraph):
		sorted_nodes = sorted(jsonGraph['node'], key=lambda node: len(node['name']), reverse=False)

		self.parse_node_hierarchy(sorted_nodes)
		self.parse_node_device(sorted_nodes)
		self.parse_node_op(sorted_nodes)
		self.parse_node_input(sorted_nodes)
		self.parse_node_attr(sorted_nodes)

		self.simplify_RDF_graph()

	def parse_node_hierarchy(self, nodes):
		for node in nodes:
			splited_data = node['name'].split('/')

			for sub_path_idx in range(len(splited_data)):

				layered_name = '/'.join(splited_data[0:sub_path_idx+1])
				# ex. loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
				# -> loss(Modal & loss) ->loss/dense_1_lost (loss & loss/dense_1_lost) -> ...

				if sub_path_idx == 0:
					self.RDF_graph.add((self.root, RDFS.member, BNode(layered_name)))
				else:
					previous_layered_name = '/'.join(splited_data[0:sub_path_idx])
					self.RDF_graph.add((BNode(previous_layered_name), RDFS.member, BNode(layered_name)))

	def parse_node_device(self, nodes):
		for node in nodes:
			if 'device'	in node.keys():
				self.RDF_graph.add((BNode(node['name']), self.on_device, Literal(node['device'])))

	def parse_node_op(self, nodes):
		for node in nodes:
			if 'op' in node.keys():
				self.RDF_graph.add((BNode(node['name']), self.has_op, Literal('op_'+node['op'])))

	def parse_node_input(self, nodes):
		for node in nodes:
			if 'input' in node.keys():
				for input_node in node['input']:
					self.RDF_graph.add((BNode(node['name']), self.has_input, BNode(input_node)))

	def parse_node_attr(self, nodes):

		for node in nodes:
			if 'attr' in node.keys():
				attr_paths = self.recursive_scan_attr(node['attr'], path=[])

				sb = BNode(node['name'])

				for attr_path in attr_paths:
					if not sum([1 if rule.parse(attr_path) else 0 for rule in self.rules]):
						print ("can not recognize %s" % str(attr_path))
						# raise Exception("Unknown attribute")

				for rule in self.rules:
					results = rule.get_results(sb)
					if results:
						self.RDF_graph += results

	def recursive_scan_attr(self, json_string, path=None):

		if path is None:
			path = []

		result = []

		if type(json_string) is dict and json_string:
			for key in json_string.keys():
				path2 = path + [key]


				if type(json_string[key]) is dict or type(json_string[key]) is list:
					result += self.recursive_scan_attr(json_string[key], path2)

				else:
					result.append((path2 + [str(json_string[key])]))

		elif type(json_string) is list and json_string:

			for idx, entity in enumerate(json_string):

				path2 = path + [str(idx)]

				if type(entity) is dict or type(entity) is list:
					result += self.recursive_scan_attr(entity, path2)

		else:
			result.append(path)

		return result

	def simplify_RDF_graph(self):

		# BFS to search nodes.
		fringe = [self.root]
		visited = []

		while fringe:
			node = fringe.pop(0)
			visited.append(node)

			for o in self.RDF_graph[node:RDFS.member]:
				if sum( [1 if word in str(o).lower() else 0 for word in self.not_interested_words]):
					pass # stop this branch of BFS is the word is interested.
				elif sum( [1 if word in str(o).lower() else 0 for word in self.interested_words]):
					visited.append(o)
				else:
					fringe.append(o)

		words_bank = visited

		for s,o in self.RDF_graph[:self.has_input]:

			found, found2 = False, False

			subjects = []
			objects = []

			for word in words_bank[::-1]:

				if self.is_member(word, s):
					subjects.append(word)
					found = True
					break

			for word in words_bank[::-1]:
				if self.is_member(word, o):
					objects.append(word)
					found2 = True
					break

			if found and found2:
				assert len(subjects) == 1 and len(objects) == 1, "wow " + s + " " + o + " " + str(subjects) + "\n" + str(objects)

				if subjects[0] != objects[0]:
					self.simplified_RDF_graph.add((subjects[0], self.has_input, objects[0]))

		for s,o in self.RDF_graph[:URIRef("http://example.org/output_shape")]:
			if s in words_bank:
				self.simplified_RDF_graph.add((s, URIRef("http://example.org/output_shape"), o))

		for s,o in self.RDF_graph[:URIRef("http://example.org/shape")]:
			if s in words_bank:
				self.simplified_RDF_graph.add((s, URIRef("http://example.org/shape"), o))

	def is_member(self, sub, obj):

		# use name information to judge, ex: "U/1/2/3/4/5" is a member of "U/1."
		s = str(sub).replace('^', '').split('/')
		o = str(obj).replace('^', '').split('/')

		for a,b in zip(s,o):
			if a != b:
				return False

		return True



def draw(func):

	def wrapper(jsonGraph, path):
		g = func(jsonGraph,path)
		colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
		G= Network(height="800px", width="70%", directed=True)
		data=[]
		for u,e,v in g[1]:
			data.append(e)

		label_encoder = LabelEncoder()
		integer_encoded = np.array(label_encoder.fit_transform(data), dtype='float')
		i=0
		for src,e,dst in g[1]:

			G.add_node(src, title=src, physics=True,color=cnames[(str(src).count('/'))], size=sizes[(str(src).count('/'))], shape="dot")
			G.add_node(dst, title=dst, physics=True,color=cnames[(str(dst).count('/'))], size=sizes[(str(dst).count('/'))], shape="dot")
			G.add_edge(src,dst,width=0.5, title=data[i],physics=True)
			i+=1

		G.hrepulsion(node_distance=120,
        			 central_gravity=0.0,
        			 spring_length=100,
        			 spring_strength=0.01,
        			 damping=0.09)
		G.show_buttons(filter_=['physics'])
		G.show(path+r"\test.html")

		# for s,p,o in g[1]:
		# 	print((s,p,o))
		return g
	return wrapper

def dump_triples(func):

	def wrapper(path):
		started_at = time.time()

		g, g1 = func(path)

		print("%s: %ss" % (func.__name__, (time.time() - started_at)) )

		for s,p,o in g:
			print("%s, %s, %s" % (s,p,o))

		for s,p,o in g1:
			print("%s, %s, %s" % (s,p,o))

		return g, g1

	return wrapper

def jsons2RDFs(path):

	for file_idx, file_path in enumerate(glob.glob("%s/*.json" % path)):

		with open(file_path) as f:
			jsonGraph = json.load(f)

			RDF_graph, s_RDF_graph = json2RDF(jsonGraph, os.path.dirname(file_path))

			RDF_graph.serialize(destination='%s/../rdf/%d.rdf' % (path, file_idx), format='turtle')
			s_RDF_graph.serialize(destination='%s/../rdf/s_%d.rdf' % (path, file_idx), format='turtle')

# @dump_triples
@draw
def json2RDF(jsonGraph, path):

	parser = Json2RDFParser()
	parser.parse(jsonGraph)

	return parser.RDF_graph, parser.simplified_RDF_graph

if __name__=="__main__"	:

	if len(sys.argv) == 2:
		path = Path(sys.argv[1])
	else:
		path = Path("..")/"tmp"/"graphs"/"json"
	path = path.resolve()

	jsons2RDFs(path)
