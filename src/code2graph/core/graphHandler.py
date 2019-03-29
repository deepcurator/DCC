'''Sess.graph and tf.Graph handler'''
from graphMETA import graphMETA
from argparse import ArgumentParser
import sys, os, json, rdflib
import tensorflow as tf
from tensorboard.backend.event_processing import plugin_event_accumulator as event_accumulator
from google.protobuf import json_format #For the json conversion
import json2graph as j2g

import shutil
from automation.dataset_utils import *
import pdb
from pathlib import Path

class graphHandler(graphMETA):
	def __init__(self,
		         graphDef  = None,
		         RDF       = None,
		         jsonGraph = None,
		         graph     = None,
		         tradGraph = None,
		         logdir    = None):
		'''Initializing the class'''
		self._graphDef  = graphDef
		self._RDF       = RDF
		self._sRDF      = None
		self._jsonGraph = jsonGraph
		self._graph     = graph
		self._tradGraph = tradGraph
		self._logdir    = logdir
		pass

	def readGraphDef(self, logdir=None):
		'''function to read graph from drive'''
		if logdir!=None:
			pass
		else:
			accumulator = event_accumulator.EventAccumulator(self._logdir)
			accumulator.Reload()
			self._graphDef = accumulator.Graph()
		return

	def printSummary(self):
		print(self.__dict__)
		self.displayGraphDef()
		return

	def displayGraphDef(self, graphDef=None):
		'''funciton to display the graph'''
		if graphDef!=None:
			#TODO: write a function to display graph
			pass
		else:
			#print(self._graph)
			for node in self._graphDef.node:
				print(node)
		return

	def convertGraphDef2Graph(self, graphDef=None):
		'''function to convert graphDef to tf.Graph()'''
		if graphDef!=None:
			self._graph = tf.import_graph_def(graphDef)
			self._graphDef = graphDef
		else:
			try:
				print(self._graphDef)
				tf.import_graph_def(self._graphDef)
				self._graph = tf.get_default_graph()
				print("Converted graphDef to tf.Graph()")
			except Exception as e:
				print("Could not get computational graph", e.args)
		return

	def convertJson2RDF(self, jsonGraph=None):
		'''function to convert json to RDF format'''
		if jsonGraph!=None:
			self._jsonGraph=jsonGraph

		self._RDF, self._sRDF = j2g.json2RDF(self._jsonGraph)
		return

	def displayRDF(self, RDF=None):
		''''funciton to display the graph'''
		if RDF!=None:
			self._RDF = RDF

		# print("completed RDF graph")
		# for s,p,o in self._RDF:
		# 	print('\t (%s) (%s) (%s)' % (str(s), str(p), str(o)))

		print("simplified RDF graph")
		for s,p,o in self._sRDF:
			print('\t (%s) (%s) (%s)' % (str(s), str(p.split('/')[-1]), str(o)))

		return


	def convertGraphDef2Json(self, graphDef=None):
		if(graphDef==None):
			self._jsonGraph = json.loads(json_format.MessageToJson(self._graphDef))
		else:
			self._jsonGraph = json.loads(json_format.MessageToJson(graphDef))
		return

	def readJson(self, logdir=None):
		'''function to read the json graph file'''
		if logdir!=None:
			self._logdir=logdir
		if self._logdir==None:
			raise ValueError('No log directory given!')
		count =len([i for i in os.listdir(self._logdir) if '.json' in i])
		with open(self._logdir+'/graph_'+str(count-1)+'.json') as f:
			self._jsonGraph = json.load(f)
		return

	def writeJson(self, jsonGraph=None, logdir=None ):
		'''function to write the json file to drive'''
		if logdir!=None:
			self._logdir=logdir
		if jsonGraph!=None:
			self._jsonGraph=jsonGraph
		if self._logdir==None:
			raise ValueError('No logdirectory given!')
		if self._jsonGraph==None:
			raise ValueError('There is no json Graph available')
		count =len([i for i in os.listdir(self._logdir) if '.json' in i])
		with open(self._logdir+'/graph_'+str(count)+'.json', 'w') as outfile:
			# self._jsonGraph=json.dumps(self._jsonGraph)
			json.dump(self._jsonGraph, outfile)


	def convertJson2GraphDef(self, jsonGraph=None):
		if(self._graphDef==None):
			self._graphDef = tf.GraphDef()
		if(jsonGraph==None):
			json_format.Parse(self._jsonGraph, self._graphDef)
		else:
			json_format.Parse(jsonGraph, self._graphDef)
		return

	def parseData(self, githubLink = None, paperLink = None):
		'''Parse the Data Given the github link and paper link
			Example usage:

			gHandle = graphHandler(logdir=os.path.expanduser(args.logdir))
			githubLink = 'https://github.com/Isnot2bad/Micro-Net'
			paperLink = 'https://arxiv.org/abs/1810.11603v1'
			gHandle.parseData(githubLink = githubLink, paperLink = paperLink)

			This will save the event file to a new folder names saved_data
			along with the paper pdf file.
		'''

		path_to_dir = os.path.abspath('./saved_data/')
		try:
			os.makedirs(path_to_dir)
		except:
			pass
		fetch_paper(paper_link = paperLink, path = path_to_dir)
		fetch_code(code_link = githubLink, path = path_to_dir)
		unzip_all(dir_path = path_to_dir)
		parseCode(parent_dir = path_to_dir)
		for path, dir_list, file_list in os.walk(path_to_dir):
			for dir in dir_list:
				shutil.rmtree(path = os.path.join(path_to_dir, dir), ignore_errors=False)
		# convertEvent2Json(logdir = path_to_dir)
		self._logdir = path_to_dir
		s = self._logdir.split('/')
		self.readGraphDef()
		self.convertGraphDef2Json()
		self.writeJson()
		# print("Json file is created for {}" .format(s[-2]))

		jsonGraph = self.readJson(logdir = path_to_dir)
		pass

	@property
	def graph(self):
		return self._graph

	@property
	def jsonGraph(self):
		return self._jsonGraph

	@property
	def RDF(self):
		return self._RDF

	@property
	def graphDef(self):
		return self._graphDef


def main():
	default_path = Path("./")/"test"/"fashion_mnist"
	default_path = default_path.resolve()

	parser = ArgumentParser(description='sess.graph/tf.graph handler')
	
	parser.add_argument('-ld','--logdir',
						default = default_path,
						type    = str,
						help    = 'directory for saved graph')
	args = parser.parse_args()

	gHandle = graphHandler(logdir=os.path.expanduser(args.logdir))

	gHandle.readGraphDef()

	gHandle.convertGraphDef2Json()

	gHandle.writeJson(logdir=str(default_path))

	gHandle.convertJson2RDF()

	gHandle.displayRDF()

if __name__=='__main__':
	main()
