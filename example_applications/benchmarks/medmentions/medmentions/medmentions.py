import json


class Mention:
    def __init__(self, doc_id, sem_type, linked_class, start, end, text):
        self.doc_id = doc_id
        self.sem_type = sem_type
        self.linked_class = linked_class
        self.start = start
        self.end = end
        self.text = text

    def __str__(self):
        return f"DOC: {self.doc_id} T: {self.sem_type} C: {self.linked_class} S: {self.start} E: {self.end} TXT: {self.text}"


class Document:
    def __init__(self, id, title, abstract, mentions: list[Mention]):
        self.id = id
        self.abstract = abstract
        self.title = title
        self.mentions = mentions

    def __str__(self):
        start_str = f"ID: {self.id}\nTitle: {self.title}\nAbstract: {self.abstract}"
        mentions_str = "\n".join([str(mention) for mention in self.mentions])
        return start_str + "\n" + mentions_str

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
