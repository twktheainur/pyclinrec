def span_tokenize(spacy, text):
    doc = spacy(text)
    tokens = []
    spans = []
    for token in doc:
        span = doc[token.i: token.i + 1]
        tokens.append(span.text)
        spans.append((span.start_char, span.end_char))
    return tokens, spans
