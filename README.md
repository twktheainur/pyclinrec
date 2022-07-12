# Python Concept Recognition Library

![PyClinRec Logo](https://raw.githubusercontent.com/twktheainur/pyclinrec/master/resources/pyclinrec_logo.png)


This package allows one to implement generic concept recognition components on the basis of a dictionary. A concept recognizer indexes the dictionaries and identifies concepts (begin and end offsets) in text. 



The current version includes two concept recognizers:

- StemIntersectionConceptRecognizer
- DoubleMetaphoneConceptRecognizer



Each recognizer implements the `ConceptRecognizer` base class and implements the `initialize`method that handles the indexing of the dictionary and `recognize` method that annotates a text with concepts from the dictionary.  `recognize`returns a list of `Annotation`objects that contain the information about the annotations. 



## Dictionary loading and formats

The `DictionaryLoader` base class allows for the implementation of dictionary loaders that accept different formats. 

Currently, a single format is supported: 

* The Mgrep TSV format (same as for BioPortal Annotator) with the `MgrepDictionaryLoader` which is the following (one per line:)

  `ID<TAB>LABEL`

  `ID` is a unique identifier for the concept (e.g. an URI)

  `LABEL` is a label for that concept (can include spaces as the separator is a tabulation)

  A concept which has several labels will result in several `ID<TAB>LABEL` lines. 

The dictionary loader can be instantiated as follows:

``````python
loader = MgrepDictionaryLoader("/path/to/tsv/file")
``````

It is fairly straightforward to implement custom dictionary loader. The loader is passed to a recognizer during its construction as will be exemplified in the next section. 



## Usage Example 

 Let us see how to instantiate a recognizer, to initialize it and to annotate a list of texts with it. Individual recognizers may require additional data files. For the two recognizers that are currently supported, the files are provided in the data directory for French (clinical text). Beware: the termination and stop lists are typically domain specific. 

```python
corpus = list() # type: List[str]
#Load some corpus as a list of strings

recognizer = IntersStemConceptRecognizer(dictionary_loader=loader,
                                         stop_words_file="pyclinrec/stopwordsfr.txt",                                 termination_terms_file="pyclinrec/termination_termsfr.txt")
recognizer.initialize()

for text in corpus: 
    annotations = recognizer.recognize(text)
    for annotation in annotations:
        concept_id = annotation.concept_id # The unique identifier of the matching concept as defined in the dictionary
        start = annotation.start # Start character offset of the annotation
        end = annotation.end # End character offset of the annotation
        matched_text = annotation.matched_text # The surface form of the text matching te annotation
        
```
