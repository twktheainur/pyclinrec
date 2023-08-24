import regex

PUNCTUATION_REGEX = regex.compile(r"[\p{C}|\p{M}|\p{P}|\p{S}]+", regex.UNICODE)
