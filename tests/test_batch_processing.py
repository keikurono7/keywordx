from keywordx.extractor import KeywordExtractor

def test_extract_many_basic():
    ke = KeywordExtractor()
    texts = [
        "Tomorrow I have a work meeting at 5pm in Bangalore.",
        "The conference is scheduled for next Monday in New York.",
        "We need to discuss the budget for the upcoming project."
    ]
    keywords = ["meeting", "time", "place", "budget"]
    results = ke.extract_many(texts, keywords)
    
    assert len(results) == 3
    assert all("semantic_matches" in r for r in results)
    assert all("entities" in r for r in results)

def test_extract_many_empty_list():
    ke = KeywordExtractor()
    texts = []
    keywords = ["meeting", "time", "place"]
    results = ke.extract_many(texts, keywords)
    
    assert len(results) == 0
    assert isinstance(results, list)

def test_extract_many_single_text():
    ke = KeywordExtractor()
    texts = ["Tomorrow I have a work meeting at 5pm in Bangalore."]
    keywords = ["meeting", "time", "place"]
    results = ke.extract_many(texts, keywords)
    
    assert len(results) == 1
    assert "semantic_matches" in results[0]
    assert "entities" in results[0]

def test_extract_many_consistency():
    ke = KeywordExtractor()
    text = "Tomorrow I have a work meeting at 5pm in Bangalore."
    keywords = ["meeting", "time", "place"]
    
    single_result = ke.extract(text, keywords)
    batch_results = ke.extract_many([text], keywords)
    
    assert len(batch_results) == 1
    assert batch_results[0] == single_result
