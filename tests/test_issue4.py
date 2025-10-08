import unittest
from keywordx import KeywordExtractor
import pprint

def entity_boosts_applied(text, keywords, weights):
    extractor = KeywordExtractor(
        entity_weights=weights
    )

    text = text
    keywords = keywords

    result = extractor.extract(text, keywords)
    return result

if __name__ == "__main__":
    pprint(entity_boosts_applied("I want to visit Paris next Friday with a budget of $1500.",["date", "place", "money"],{
            'GPE': 1.5,
            'DATE': 1.3,
            'MONEY': 0.8   
        }))
