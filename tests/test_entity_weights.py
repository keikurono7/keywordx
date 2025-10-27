import pytest
from keywordx import KeywordExtractor

def test_custom_entity_weights():
    # Test pure entity matches (no semantic overlap)
    ke = KeywordExtractor(entity_weights={
        'DATE': 1.5,
        'GPE': 1.2,
        'TIME': 0.8
    })
    
    # Use a text where keywords don't have semantic matches
    text = "Meeting next Friday at 3pm in London."
    keywords = ["event_date", "location", "meeting_time"]  # No direct matches
    
    # Map these to our entity types
    entity_map = {"event_date": "date", "location": "place", "meeting_time": "time"}
    
    result = ke.extract(text, keywords)
    
    # Check if entities are present
    assert len(result["entities"]) >= 3
    
    # Convert semantic matches to dict for easier testing
    matches = {m["keyword"]: m for m in result["semantic_matches"]}
    
    # For pure entity matches, scores should be base_score * boost
    assert len(matches) <= 3  # We should only get entity matches

def test_invalid_entity_weights():
    # Test with invalid entity type
    with pytest.raises(ValueError) as exc_info:
        KeywordExtractor(entity_weights={'INVALID': 1.5})
    assert "Invalid entity types" in str(exc_info.value)
    
    # Test with invalid type
    with pytest.raises(TypeError) as exc_info:
        KeywordExtractor(entity_weights=[1, 2, 3])
    assert "must be a dict" in str(exc_info.value)

def test_default_weights():
    # Test that semantic matches win over unweighted entity matches
    ke = KeywordExtractor()  # No custom weights
    
    # Use text with both semantic and entity matches
    text = "Important meeting with John tomorrow at noon in London"
    keywords = ["meeting", "event_time", "location"]
    
    result = ke.extract(text, keywords)
    matches = {m["keyword"]: m for m in result["semantic_matches"]}
    
    # meeting should have high semantic match
    assert matches["meeting"]["score"] > 0.8  # Strong semantic match
    
    # entity matches should use base score
    assert 0.5 <= matches.get("location", {"score": 0})["score"] <= 0.7  # Around base score

def test_score_multiplication():
    # Test that entity weights multiply existing semantic scores
    ke = KeywordExtractor(entity_weights={
        'DATE': 1.5,
        'GPE': 1.2
    })
    
    # Use a text where "meeting" has both semantic and entity matches
    text = "The annual meeting is scheduled for next Friday in Paris."
    keywords = ["event", "date", "location"]
    
    result = ke.extract(text, keywords)
    matches = {m["keyword"]: m for m in result["semantic_matches"]}
    
    # Check that scores are properly boosted
    assert matches["date"]["score"] == 0.6 * 1.5  # Entity-only with boost
    assert matches["location"]["score"] == 0.6 * 1.2  # Entity-only with boost