# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:08:00 2019

@author: z003z47y
"""

from xml.dom import minidom
import glob
import pandas as pd
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from IPython.display import display
import pandas as pd
import json
import numpy as np
from pandas.io.json import json_normalize


def execute_query(sparqlQuery):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(sparqlQuery)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = json_normalize(results["results"]["bindings"])
#    results_df = results_df[results_df['o.type'] == "uri"]
#    results_df = results_df.drop(['o.datatype','o.type', 'o.xml:lang','p.type'],axis=1)
#    results_df = results_df[['p.value','o.value','countS.value']]
#    results_df['countS.value'] = results_df['countS.value'].astype(int)
#    results_df = results_df[results_df['countS.value'] > 1]
    return results_df

newdf = pd.read_pickle('../../paperswithcode/txt/annotated/annotated.pkl')

annotation = newdf.loc[0,'Annotated']
print(annotation)

xmlString = '<data>  <items> <item name="item1">item1abc</item> <item name="item2">item2abc</item></items></data> '

# parse an xml file by name
mydoc = minidom.parseString(annotation)

items = mydoc.getElementsByTagName('Resource')

# one specific item attribute
print('Item #2 attribute:')  
print(items[1].attributes['URI'].value)
print(items[1].attributes['surfaceForm'].value)
print(items[1].attributes['offset'].value)

print(items[1].attributes['surfaceForm'].value + " isA "  +  items[1].attributes['URI'].value )


queryStart = """ PREFIX  schema: <http://schema.org/>
PREFIX  dbr:  <http://dbpedia.org/resource/>
PREFIX  umbel-rc: <http://umbel.org/umbel/rc/>
PREFIX  dbc:  <http://dbpedia.org/resource/Category:>
PREFIX  owl:  <http://www.w3.org/2002/07/owl#>
PREFIX  yago: <http://dbpedia.org/class/yago/>
PREFIX  rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX  dbo:  <http://dbpedia.org/ontology/>
PREFIX  dbp:  <http://dbpedia.org/property/>
PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX  dcterms: <http://purl.org/dc/terms/>
PREFIX  dbpedia-wikidata: <http://wikidata.dbpedia.org/resource/>
PREFIX  dul:  <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX  foaf: <http://xmlns.com/foaf/0.1/>
PREFIX  dc:   <http://purl.org/dc/elements/1.1/>
PREFIX prov: <http://www.w3.org/ns/prov#>

SELECT   ?o 
WHERE {  <"""

queryEnd = """  rdf:type ?o .
}"""

query = queryStart + items[1].attributes['URI'].value + ">" + queryEnd

results_df = execute_query(query)

# all item attributes
print('\nAll attributes:')  
for elem in items:  
    print(elem.attributes['URI'].value)

# one specific item's data
print('\nItem #2 data:')  
print(items[1].firstChild.data)  
print(items[1].childNodes[0].data)

# all items data
print('\nAll item data:')  
for elem in items:  
    print(elem.firstChild.data)