# Instructions

## 1. Install requirements

```shell
pip install -r requirements.txt
```

## 2. Generate dictionary

```shell
python generate_dictionary.py http://my.endpoint.com/sparql --from http://my.named.graph/ -l en -o agrovoc_dictionary.tsv 
```
The `FROM` clause support is experimental, please report any "unexpected features". 

## 3. Instanciate annotator and annotate

```python
from annotator import AgrovocAnnotator

annotator = AgrovocAnnotator("path/to/dictionary.tsv", language="en")

annotations = annotator.annotate("This is my text")
```

The annotator expects to find the termination term and stop word files in `data/` relative to itself.

The `annotations` list contains `Annotation` objects, containing the following attributes:

- `start`: start offset of the annotation
- `end` : end offset of the annotation
- `matched_text`: surface text matched
- `matched_length`: length of the matched text
- `concept`: The corresponding `Concept`, has a `uri`, a list of `labels`, optionally a `definition`
- `confidence_score`: The confidence score for the annotation when possible
