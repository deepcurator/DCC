# Alignment of Annotation and Ontologies

## CSO Processing

[Computer Science Ontology](https://cso.kmi.open.ac.uk/home) is a large-scale ontology of research areas in Computer Science, containing multiple entities and instances of relations. 
We aimed to integrate CSO into our knowledge graph and to use occurrences of entities and relations from CSO in text as valid annotations. We explore CSO owl file:
- It contains ~162K triples
- There are multiple entities, some pointing to external resources such as Wikipedia, DBPedia, microsoft academia
- There are 8 unique relations:
  rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
  rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#label')
  rdflib.term.URIRef('http://cso.kmi.open.ac.uk/schema/cso#contributesTo') - similar to our 'usedFor'…
  rdflib.term.URIRef('http://cso.kmi.open.ac.uk/schema/cso#relatedEquivalent') – equivalent to our so 'sameAs'
  rdflib.term.URIRef('http://cso.kmi.open.ac.uk/schema/cso#preferentialEquivalent') – equivalent to our 'sameAs'. Moreover, the entry pointed to is the one usually containing all the relevant links and relations
  rdflib.term.URIRef('http://www.w3.org/2002/07/owl#sameAs') – refers only to external URIs (DBPedia, Wikipedia)
  rdflib.term.URIRef('http://cso.kmi.open.ac.uk/schema/cso#superTopicOf') – this is inverse of our 'isA' relation
  rdflib.term.URIRef('http://schema.org/relatedLink')

Further analysis and attempts to build NER & RE using this ontology revealed a number of difficulties mapping topic names (after decoding from uri to utf8) since they often include terms in parenthesis:
- abbreviations (or reverse) in inconsistent styles/formats, ex:
  'xml (extensible_markup_language)' – abbreviation is outside of parenthesis, while the full term is inside
  'support_vector_machine (svms)' – mismatch in plural vs singular
  'peer to peer (p2p) network' – abbreviation is in middle of phrase, use of '2' to represent '2'
  'orthogonal frequency division multiple access (ofd' –  incomplete parenthesis
- certain hard-to-detect alternatives, ex: 'two-dimensional (2d)', 'wireless (wifi) communication'
- contextual terms, ex: 'android (operating system)', 'prolog (programming language)' - those need to be removed to identify entities properly
- complex terms, ex: '(min ,max ,+) functions','(i ,j) conditions' – here the values in parenthesis are part of topic name and can't be removed. So this and previous case cannot be separated.


Therefore, the following approach is taken to leverage CSO:
- Processing topics:
	- We focus on nodes with uri starting with ''https://cso.kmi.open.ac.uk/topics/'
	- In the URI suffix (after topic), we replace '_' with spaces
	- Check for parenthesis and attempt to verify whether inner part of text is abbreviation of outer (or vice-versa).
	- Check the shorter part against common stopwords, to avoid having abbreviations like 'of' causing problems later
	- If yes, we separate the values, and add 'sameAs' relation between them
	- If no, we ignore it - it won't be very helpful in entity detection anyway
- Processing relations:
	- In 1st pass we create a map for each uri to 'preferentialEquivalent'. If a uri does not have a 'preferentialEquivalent', it maps to itself.
	- Then, for each entity string, we create a map for the string to the full uri of the preferential equivalent
	- We keep a map from pairs of strings to relations of interest: cso#contributesTo, superTopicOf

This is implemented in script [collect_cso.py](collect_cso.py). 


## Our Annotations

In order to train and evaluate NER and RE models, we need ground truth annotations, or, at least, a reasonably high-quality annotations. Human annotators are costly and inconsistent, so we'd like to minimize their role.  We'd also like to be able to produce initial annotations for a new piece of text. For this we need an ontology:
- a vocabulary of strings mapping to entity types of interest
- a mapping from pairs of strings to relations of interest
The first source of such information at our disposal is the set of annotations we produced during Phase 1 on 100 abstracts. 
We collect all this data with some automated clean-up, and manual review. This results in:
- 1121 terms mapped to entities 
- 698 mappings from term pairs to relations 
This information is collected by a script [collect_brat_vocabulary.py](collect_brat_vocabulary.py). 
We expand this set with UWa annotation described nextwould

## UWa Annotations

There are several differences between our annotations and UWa:
- Entities: ours (6): Method, Generic, Task, Material, Eval, Other  – UWa (6): Method, Generic, Task, Material, Metric, OtherScientificTerms
- Relations: ours (7): Compare, Conjunction, Feature-of, Part-of, Used-for, IsA, sameAs – UWa (8): USED-FOR, FEATURE-OF, COREF, CONJUNCTION, HYPONYM-OF, EVALUATE-FOR, PART-OF, COMPARE]

These can be resolved by mapping UWa entities and relations to ours as follows:
- Metric -> Eval
- OtherScientificTerms -> Other
- Hyponym -> isA
- Also, we remove all COREF relations because we currently do not handle them in the same way
- All other entities and relations stay the same (up to case change)
The original UWa annotations contains 5885 distinct entity strings and 6035 relations. First, we convert everything to lower-case and remove duplicates: 5813 entities and 6008 relations.  After removing 'Coref', only 4659 relations are left. Then, we remove 'conflict' - when same string has two entity types, or same pair has two relation types. This results in:  5539 entities and 4647 relations.

This information is collected by a script [collect_uwa_vocabulary.py](collect_uwa_vocabulary.py).

Further we combine our and UWA annotations, before merging with CSO (next section). We use a similar approach to the above cleaning: we concatenate data frames of entities and relations from our annotation and UWA, remove duplicate entries and then remove conflicts (of which there are relatively few). This leaves us with 6512 entities and 5342 relations.

## Merging CSO and Human Annotations

While CSO appeared to be a promising ontology, we found that it is not as useful as expected: the only entity type there is 'topic' and the set of relations is limited and applies to topics rather than entities (ex. 'neural networks' may be superTopicOf 'backpropagation', but this clearly 'backpropagations is not a type of 'neural network'). However, we realized we can attempt to expand our vocabulary and set of relations by combining CSO and our annotations. The process is as follows:

1. We extract topics from CSO as strings, mapping them to URIs
2. We also are able to extract some useful relations from CSO corresponding to: usedFor and sameAs.  Details for these first two steps are at CSO Analysis, and it is performed by a script called collect_cso.py
3. We now find which of CSO terms occur in our annotation data - and generate sets of terms both from CSO and annotations that we can thus map to the same URI.
4. We create new URIs for annotation terms that didn't match any in CSO
5. For all the URIs from steps 3 and 4 we can create a mapping to our entity types (Method, Data, Generic) based on annotations
6. We can merge relations from CSO and from our annotations 
7. Thus we end up with:
	mapping from strings to URIs, with multiple strings mapping to same URI: 20102 strings  (though many may not be relevant for deep learning)
	mapping from URIs to Entity Types: 6455 URIs
	mapping from pairs of URIs to Relations: 49918 pairs
Steps 3-7 are performed by [merge_cso_brat.py](merge_cso_brat.py)

Note that the above process may result in conflicts between cso and annotations, or between annotations:


##Automatic Annotation
We can now use the data structures created to annotate new pieces of text. Specifically, given some text we can:
1. Find occurrences of all strings in our vocabulary in the text, with corresponding URIs
2. Select those for which we have URI-to-entity mapping: we now have performed entity recognition
3. For all pairs of strings matching entities check for relations in the URI pairs-to-relations map: this provides initial relation extraction
4. This workflow is implemented in: auto_annotate.py

This functionality is implemented in [auto_annotator.py](auto_annotator.py)

## Next Steps
The main improvements are to be gained from improving the quality of the data collected:

- Improved resolution of conflicts in manual annotations - possibly with a human review of conflicts, which are relatively rare
- Improved resolution of conflicts when merging manual annotations with CSO, also possibly with human review
- Improved relation extraction by creating an explicit graph from the 'ground truth' data and performing reasoning about relations. For example, if A Used-For B and B isA C, then we should be able to infer that A isUsedFor C without this being explicitly in the relation map.
- Expanding vocabulary and relations by bringing in other ontologies, such as DBPedia and Yago - it is necessary to keep in mind however that these will not have the same entity types, and many of the relations we are looking for, and that only a fraction of such ontologies will be relevant for our domains. Thus the anticipated effort will need to be balanced by expected benefits.






