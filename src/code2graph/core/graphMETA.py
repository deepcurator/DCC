'''Abstract class for defining the sess.graph handler.'''
from abc import ABCMeta

class graphMETA:
	__metaclass__ = ABCMeta

	def __init__(self,
		         graphDef  = None,
		         RDF       = None,
		         jsonGraph = None,
		         graph     = None,
		         tradGraph = None,
		         logdir    = None):
		'''Initializing the class'''
		pass

	def readGraphDef(self, logdir=None):
		'''function to read GraphDef from drive'''
		pass

	def writeGraphDef(self, graphDef=None, logdir=None):
		'''function to write the GraphDef to drive'''
		pass

	def displayGraphDef(self, graphDef=None):
		'''function to display the graphDef'''
		pass

	def readRDF(self, logdir=None):
		'''function to read the RDF file'''
		pass

	def writeRDF(self, RDF=None, logdir=None ):
		'''function to write the RDF file to drive'''
		pass

	def displayRDF(self, RDF=None):
		'''function to display the RDF'''
		pass

	def readJson(self, logdir=None):
		'''function to read the json graph file'''
		pass

	def writeJson(self, jsonGraph=None, logdir=None ):
		'''function to write the json file to drive'''
		pass

	def displayJson(self, jsonGraph=None):
		'''function to display the Json Graph'''
		pass

	def convertGraphDef2Json(self, graphDef=None):
		'''function to convert GraphDef to Json format'''
		pass

	def convertJson2GraphDef(self, jsonGraph=None):
		'''function to convert json graph to GraphDef format'''
		pass

	def convertJson2RDF(self, jsonGraph=None):
		'''function to convert json to RDF format'''
		pass

	def convertRDF2Json(self, RDF=None):
		'''function to convert RDF to JsonGraph'''
		pass

	def convertGraphDef2Graph(self, graphDef=None):
		'''function to convert graphDef to tf.Graph()'''
		pass

	def convertGraph2GraphDef(self, graph=None):
		'''function to convert tf.Graph() to tf.GraphDef()'''
		pass

	def convertjson2tradGraph(self, graph=None):
		'''function to convert json  to tradional graph with V,E'''
		pass

	def rdfBinder(self, rdf=None):
		'''Bind the generated RDF with tensorflow library types
		to convert it back to code template'''
		pass

	def printSummary(self):
		'''display summary of graphhandle'''
		pass

	def parseData(self, githubLink = None, paperLink = None):
		'''Parse the Data Given the github link and paper link'''
		pass
