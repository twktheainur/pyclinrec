def span_tokenize(spacy, text):
    doc = spacy(text)
    tokens = []
    spans = []
    for token in doc:
        span = doc[token.i: token.i + 1]
        tokens.append(span.text)
        spans.append((span.start_char, span.end_char))
    return tokens, spans


def span_tokenize_fast(tokenizer, text):
    """Tokenize using only spacy's tokenizer (no NLP pipeline).

    This is ~10-50x faster than span_tokenize() which runs the full pipeline.
    Use this when only token text and positions are needed (e.g., dictionary indexing).
    """
    doc = tokenizer(text)
    tokens = []
    spans = []
    for token in doc:
        tokens.append(token.text)
        spans.append((token.idx, token.idx + len(token.text)))
    return tokens, spans


def batch_span_tokenize_fast(tokenizer, texts, batch_size=1000):
    """Batch-tokenize multiple texts using only spacy's tokenizer.

    Returns a list of (tokens, spans) tuples, one per input text.
    Uses tokenizer.pipe() for efficient batch processing.
    """
    results = []
    for doc in tokenizer.pipe(texts, batch_size=batch_size):
        tokens = []
        spans = []
        for token in doc:
            tokens.append(token.text)
            spans.append((token.idx, token.idx + len(token.text)))
        results.append((tokens, spans))
    return results
