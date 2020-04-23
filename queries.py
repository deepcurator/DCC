from SPARQLWrapper import SPARQLWrapper, JSON
from IPython.display import display
import pandas as pd
import json
import numpy as np
from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

pd.options.display.max_colwidth = 100
pd.options.display.max_rows = 999


def execute_query(sparqlQuery):
    sparql = SPARQLWrapper("http://localhost:8082/sparql")
    sparql.setQuery(sparqlQuery)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = json_normalize(results["results"]["bindings"])
    # results_df = results_df[results_df['o.type'] == "uri"]
    # results_df = results_df.drop(['o.datatype','o.type', 'o.xml:lang','p.type'],axis=1)
    # results_df = results_df[['p.value','o.value','countS.value']]
    # results_df['countS.value'] = results_df['countS.value'].astype(int)
    # results_df = results_df[results_df['countS.value'] > 1]
    return results_df



### Number of publications by year across all conferences
yearCount = """
Select count(distinct ?s) as ?Count ?year  where {


?s a <https://github.com/deepcurator/DCC/Publication> .
?s <https://github.com/deepcurator/DCC/conferenceSeries> ?o .
?s <https://github.com/deepcurator/DCC/yearOfPublication> ?year .


}Group by ?year ORDER by DESC(?year)


"""


results_df = execute_query(yearCount)
results_df = results_df.drop(['Count.datatype', 'Count.type',  'year.datatype',
       'year.type'],axis=1)
results_df.columns
results_df.head()


results_df = results_df.astype(int)


ax = results_df.plot(kind='bar',x='year.value',y='Count.value',color='blue')
ax.set(xlabel = "Years", ylabel = "Publication Counts")

#### Number of publications by conference series and by year

conferenceyear = """

Select count(distinct ?s) as ?Count ?conference ?year  where {


?s a <https://github.com/deepcurator/DCC/Publication> .
?s <https://github.com/deepcurator/DCC/conferenceSeries> ?conference .
?s <https://github.com/deepcurator/DCC/yearOfPublication> ?year .


}

"""
results_df = execute_query(conferenceyear)
results_df.columns
results_df = results_df.drop(['Count.datatype', 'Count.type',  'conference.type','year.datatype',
       'year.type'],axis=1)
results_df.head()
results_df["Count.value"] = pd.to_numeric(results_df["Count.value"])
results_df["year.value"] = pd.to_numeric(results_df["year.value"])
results_df.groupby(['conference.value','year.value']).size().unstack().plot(kind='bar',stacked='True')
plt.show()


### Year and platform
platformyear = """

Select count(?platform) as ?countplatform ?platform ?year where {


?s a <https://github.com/deepcurator/DCC/Publication> .
?s <https://github.com/deepcurator/DCC/conferenceSeries> ?conference .
?s <https://github.com/deepcurator/DCC/yearOfPublication> ?year .
?s <https://github.com/deepcurator/DCC/platform> ?platform .

} 
group by  ?platform ?year order by DESC(?year)

"""

results_df = execute_query(platformyear)
results_df.columns
results_df = results_df.drop(['countplatform.datatype', 'countplatform.type', 'platform.type' ,'year.datatype','year.type'],axis=1)
results_df.head()
results_df.groupby(['platform.value','year.value']).size().unstack().plot(kind='bar',stacked='True')
plt.show()


#### Trends in year for pytorch
platformtrends = """

Select count(?platform) as ?pytorch ?platform ?year where {


?s a <https://github.com/deepcurator/DCC/Publication> .
?s <https://github.com/deepcurator/DCC/conferenceSeries> ?conference .
?s <https://github.com/deepcurator/DCC/yearOfPublication> ?year .
?s <https://github.com/deepcurator/DCC/platform> ?platform .

FILTER(STR(?platform) ="pytorch")
} 
group by  ?platform ?year order by DESC(?year)

"""

results_df = execute_query(platformtrends)
results_df.columns
results_df.head()
results_df = results_df.drop(['pytorch.datatype', 'platform.type','pytorch.type',  'platform.type','year.datatype',
       'year.type'],axis=1)
results_df.head()
results_df["pytorch.value"] = pd.to_numeric(results_df["pytorch.value"])
results_df["year.value"] = pd.to_numeric(results_df["year.value"])
# results_df.groupby(['platform.value','year.value']).size().unstack().plot(kind='bar',stacked='True')

ax = results_df.plot(kind='bar',x='year.value',y='pytorch.value',color='blue')
ax.set(xlabel = "Years", ylabel = "Platform")
# results_df["Count.value"] = pd.to_numeric(results_df["Count.value"])
# results_df["year.value"] = pd.to_numeric(results_df["year.value"])
# plt.show()


#### Trends in year for tensorflow
platformtrends = """

Select count(?platform) as ?tensorflow ?platform ?year where {


?s a <https://github.com/deepcurator/DCC/Publication> .
?s <https://github.com/deepcurator/DCC/conferenceSeries> ?conference .
?s <https://github.com/deepcurator/DCC/yearOfPublication> ?year .
?s <https://github.com/deepcurator/DCC/platform> ?platform .

FILTER(STR(?platform) ="tensorflow")
} 
group by  ?platform ?year order by DESC(?year)

"""

results_df = execute_query(platformtrends)
results_df.columns
results_df.head()
results_df = results_df.drop(['tensorflow.datatype', 'platform.type','tensorflow.type',  'platform.type','year.datatype',
       'year.type'],axis=1)
results_df.head()
results_df["tensorflow.value"] = pd.to_numeric(results_df["tensorflow.value"])
results_df["year.value"] = pd.to_numeric(results_df["year.value"])
# results_df.groupby(['platform.value','year.value']).size().unstack().plot(kind='bar',stacked='True')

ax = results_df.plot(kind='bar',x='year.value',y='tensorflow.value',color='blue')
ax.set(xlabel = "Years", ylabel = "Platform")


#### Function trends across conferences 

functiontrends = """

Select count(?type) as ?counttype  ?type  ?conference where {

?s <https://github.com/deepcurator/DCC/conferenceSeries> ?conference .
?s <https://github.com/deepcurator/DCC/yearOfPublication> ?year .
?s <https://github.com/deepcurator/DCC/hasRepository> ?repository .
?repository <https://github.com/deepcurator/DCC/hasFunction> ?y.
?y a ?type .
FILTER(!(STR(?type) = "https://github.com/deepcurator/DCC/UserDefined")).

}group by ?type ?conference ORDER by DESC(?counttype)


"""

results_df = execute_query(functiontrends)
results_df.columns

results_df = results_df.drop(['counttype.datatype', 'conference.type','counttype.type',  'type.type',
       ],axis=1)
results_df.head()
results_df["counttype.value"] = pd.to_numeric(results_df["counttype.value"])
results_df = results_df[results_df['counttype.value'] > 1000]
# results_df["year.value"] = pd.to_numeric(results_df["year.value"])
# results_df.groupby(['type.value','counttype.value']).size().unstack().plot(kind='bar',stacked='True',legend='False')



# Second plot
ax = results_df.plot(kind='bar',x='type.value',y='counttype.value',color='blue')
ax.set(xlabel = "TF Functions", ylabel = "Count")


#### Function trends across years

functionyeartrends = """

Select count(?type) as ?counttype  ?type  ?year where {

?s <https://github.com/deepcurator/DCC/conferenceSeries> ?conference .
?s <https://github.com/deepcurator/DCC/yearOfPublication> ?year .
?s <https://github.com/deepcurator/DCC/hasRepository> ?repository .
?repository <https://github.com/deepcurator/DCC/hasFunction> ?y.
?y a ?type .
FILTER(!(STR(?type) = "https://github.com/deepcurator/DCC/UserDefined")).

}group by ?type ?year ORDER by DESC(?counttype)


"""

results_df = execute_query(functionyeartrends)
results_df.columns

results_df = results_df.drop(['counttype.datatype', 'year.type','counttype.type',  'type.type',
       ],axis=1)
results_df.head()
results_df["counttype.value"] = pd.to_numeric(results_df["counttype.value"])
results_df = results_df[results_df['counttype.value'] > 1000]
# results_df["year.value"] = pd.to_numeric(results_df["year.value"])
# results_df.groupby(['counttype.value','year.value']).size().unstack().plot(kind='bar',stacked='True',legend='False')

ax = results_df.plot(kind='bar',x='type.value',y='counttype.value',color='blue')
ax.set(xlabel = "TF Functions", ylabel = "Count")



## CSO Queries :
# The first query shows all the CSO objects and the different modalities
# CSO concepts are linked to image and text modalities
csoquery1 = """

Select  ?cso  ?type where {

?o <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?cso .
?o a ?type .

}

"""
results_df = execute_query(csoquery1)
results_df.columns

results_df = results_df.drop(['cso.type', 'type.type'],axis=1)
results_df.head()




## Queries that shows publications and  CSO entities for modalities
## Deep dive into the CSO text
csoquery2 = """   


Select ?publication count(?cso) as ?countcso  where {

?publication a <https://github.com/deepcurator/DCC/Publication> .
?publication <https://github.com/deepcurator/DCC/hasEntity> ?entity .
?entity <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?cso .

}order by DESC(?countcso)

"""

results_df = execute_query(csoquery2)
results_df.columns

results_df = results_df.drop(['countcso.datatype', 'countcso.type','publication.type'],axis=1)
results_df.head()

csoimagequery = """ 
Select distinct ?publication ?cso  where {



?publication <https://github.com/deepcurator/DCC/hasFigure> ?f .

?component <https://github.com/deepcurator/DCC/partOf> ?f .
?component <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?cso.


}
"""
results_df = execute_query(csoimagequery)
results_df.columns

results_df = results_df.drop(['cso.type', 'publication.type'],axis=1)
results_df.head()



## Query that shows types and CSO objects

csoquery3 = """

Select ?type ?cso  where {

?s a <https://github.com/deepcurator/DCC/Publication> .
?s <https://github.com/deepcurator/DCC/hasEntity> ?o .
?o a ?type .
?o <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?cso .

}

"""

results_df = execute_query(csoquery3)
results_df.columns

results_df = results_df.drop(['cso.type', 'publication.type'],axis=1)
results_df.head()


# Given a CSO topic, how are many of them are aligned to the text and image entities

csoquery4  = """

Select distinct ?type count(?cso) as ?csocount where {

?s a <https://github.com/deepcurator/DCC/Publication> .
?s <https://github.com/deepcurator/DCC/hasEntity> ?o .
?o a ?type .
?o <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?cso .

}order by desc(?csocount)

"""


results_df = execute_query(csoquery4)
results_df.columns

results_df = results_df.drop(['csocount.datatype', 'csocount.type','type.type'],axis=1)
results_df.head()


## Given a particular machine learning task, what datasets would you recommend?

task_dataset_query  = """

Select  distinct ?task ?dataset where {


?publication <https://github.com/deepcurator/DCC/hasEntity> ?taskentity .
?publication <https://github.com/deepcurator/DCC/hasEntity> ?dataset.

?taskentity a <https://github.com/deepcurator/DCC/Task> .
?dataset a <https://github.com/deepcurator/DCC/Material> .

?taskentity <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?task .

}group by ?task ?dataset

"""


results_df = execute_query(task_dataset_query)
results_df.columns

results_df = results_df.drop(['dataset.type', 'task.type'],axis=1)
results_df.head()


## Top ML methods for tasks from the Knowledge graph
## They show the cso equivalents

method_task_query = """ 

Select  distinct ?method ?task where {


?publication <https://github.com/deepcurator/DCC/hasEntity> ?methodentity .
?publication <https://github.com/deepcurator/DCC/hasEntity> ?taskentity.

?taskentity a <https://github.com/deepcurator/DCC/Task> .
?methodentity a <https://github.com/deepcurator/DCC/Method> .

?methodentity <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?method .
?taskentity <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?task .

}group by ?method ?task


"""

results_df = execute_query(method_task_query)
results_df.columns

results_df = results_df.drop(['method.type', 'task.type'],axis=1)
results_df.head()


### Top DL methods and the associated tensorflow functions used for implementation

method_tf_query = """

Select count(?type) as ?counttype ?method ?type  where {

?s <https://github.com/deepcurator/DCC/hasEntity> ?methodentity .
?methodentity a <https://github.com/deepcurator/DCC/Method> .
?methodentity <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?method .
?s <https://github.com/deepcurator/DCC/hasRepository> ?repository .
?repository <https://github.com/deepcurator/DCC/hasFunction> ?y.

?y a ?type .
FILTER(!(STR(?type) = "https://github.com/deepcurator/DCC/UserDefined")).

}group by ?method ?type ORDER by DESC(?counttype)



"""

results_df = execute_query(method_tf_query)
results_df.columns

results_df = results_df.drop(['counttype.datatype','counttype.type','method.type', 'type.type'],axis=1)
results_df.head()

### Top DL tasks and the associated tensorflow functions used for implementation

task_tf_query = """

Select count(?type) as ?counttype ?task ?type  where {

?s <https://github.com/deepcurator/DCC/hasEntity> ?taskentity .
?taskentity <https://github.com/deepcurator/DCC/hasCSOEquivalent> ?task .
?taskentity a <https://github.com/deepcurator/DCC/Task> .
?s <https://github.com/deepcurator/DCC/hasRepository> ?repository .
?repository <https://github.com/deepcurator/DCC/hasFunction> ?y.

?y a ?type .
FILTER(!(STR(?type) = "https://github.com/deepcurator/DCC/UserDefined")).

}group by ?task ?type ORDER by DESC(?counttype)


"""

results_df = execute_query(task_tf_query)
results_df.columns

results_df = results_df.drop(['counttype.datatype','counttype.type','task.type', 'type.type'],axis=1)
results_df.head()