"""
Example: Batch Processing with extract_many()

This example demonstrates how to use the extract_many() method
to process multiple documents efficiently.
"""
from keywordx import KeywordExtractor

ke = KeywordExtractor()

texts = [
    "Tomorrow I have a work meeting at 5pm in Bangalore.",
    "The conference is scheduled for next Monday in New York at 2pm.",
    "We need to discuss the budget for the upcoming project in Paris.",
    "The annual review meeting happens every December in London.",
]

keywords = ["meeting", "time", "place", "budget"]

print("Processing multiple documents with extract_many()...\n")
results = ke.extract_many(texts, keywords)

for i, (text, result) in enumerate(zip(texts, results), 1):
    print(f"Document {i}:")
    print(f"  Text: {text}")
    print(f"  Matches: {len(result['semantic_matches'])}")
    for match in result['semantic_matches']:
        print(f"    - {match['keyword']}: {match['match']} (score: {match['score']:.3f})")
    print(f"  Entities: {len(result['entities'])}")
    for entity in result['entities']:
        print(f"    - {entity['type']}: {entity['text']}")
    print()
