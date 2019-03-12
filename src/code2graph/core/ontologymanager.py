from scraper_tf_voc import TFVocScraper
from rdflib import Graph, BNode, RDFS, RDF, URIRef, Literal

class OntologyManager:

	is_type = BNode("is_type")

	def __init__(self):
		self.scraper = TFVocScraper("r1.13")	
		
		self.tf_types = self.scraper.root

		self.type_hash = {}
		self.recur_build_hash(self.tf_types)
		
		self.is_type = BNode("is_type")

	def recur_build_hash(self, node):
		
		if "children" not in node: 
			self.type_hash[node['name']] = node
		
		elif "children" in node:
			if node['name'] == 'tf.logging' or \
			   node['name'] == 'tf.io' or \
			   node['name'] == 'tf.strings' or \
			   node['name'] == 'tf.gfile': 
			   # filtering some non-architecture components.
				return
			
			for child in node['children']:
				self.recur_build_hash(child)

	# exact search
	def exact_search(self, key):
		results = []

		for hashkey in self.type_hash:
			comp = hashkey.split('.')[-1]

			if comp == key: 
				results.append(hashkey)

		return results

	# fuzzy search 
	def fuzzy_search(self, key):
		keys = key.split('.')

		scores = {}

		for hashmap in self.type_hash.keys():
			score = 0
			hashmaps = hashmap.split('.')
			
			for idx_x, x in enumerate(keys):
				for idx_y, y in enumerate(hashmaps):
					if x == y:
						score+= (6.0*(1+idx_x) / (len(keys)*len(hashmaps)))
					elif x == y.lower():
						score+= (3.0*(1+idx_x) / (len(keys)*len(hashmaps)))
					elif x.lower() == y.lower():
						score+= (3.0*(1+idx_x) / (len(keys)*len(hashmaps)))
					elif x in y:
						score+= (1.0*(1+idx_x) / (len(keys)*len(hashmaps)))
					elif x in y.lower():
						score+= (1.0*(1+idx_x) / (len(keys)*len(hashmaps)))
					elif x.lower() in y.lower():
						score+= (1.0*(1+idx_x) / (len(keys)*len(hashmaps)))

			if score > 1.5:
				scores[hashmap] = score

		sorted_by_value = sorted(scores.items(), key=lambda kv: -kv[1])
		sorted_scores = dict(sorted_by_value)
		return_list = list(sorted_scores.keys())

		return return_list
	
if __name__ == "__main__":
	
	ontology_manager = OntologyManager()

	print("Exact Search:")
	print(ontology_manager.exact_search("Dense"))
	print(ontology_manager.exact_search("dense"))
	print(ontology_manager.exact_search("relu"))
	print(ontology_manager.exact_search("crelu"))
	print(ontology_manager.exact_search("exp"))

	print("Fuzzy Search:")
	print(ontology_manager.fuzzy_search("Dense"))
	print(ontology_manager.fuzzy_search("dense"))
	print(ontology_manager.fuzzy_search("relu"))
	print(ontology_manager.fuzzy_search("crelu"))
	print(ontology_manager.fuzzy_search("exp"))


