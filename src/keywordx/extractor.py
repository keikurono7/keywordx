from .chunker import chunk_phrases
from .embeddings import embed_texts, whiten
from .matcher import score_matches
from .ner import extract_structured
from .utils import load_spacy_model
from collections.abc import Mapping
from numbers import Real
from typing import Optional


class KeywordExtractor:
    def __init__(self, baseline_text="is the a", entity_weights: Optional[Mapping] = None):
        """
        Args:
            baseline_text (str): Text used for embedding normalization.
            entity_weights (dict): Custom boost weights for entity types.
        Example: {'DATE': 1.5, 'GPE': 1.2, 'MONEY': 0.8}
        """
        self.baseline_text = baseline_text
        self._load_model()
        self.VALID_ENTITY_TYPES = {i for i in self.model.get_pipe("ner").labels}
        
        # Include special internal entity that may be added by ner parsing
        self.VALID_ENTITY_TYPES.add("PARSED_DATE")

        if entity_weights is not None and not isinstance(entity_weights, Mapping):
            raise TypeError(f"entity_weights must be a dict, not {type(entity_weights).__name__}")

        if entity_weights:
            invalid_keys = [k for k in entity_weights if k not in self.VALID_ENTITY_TYPES]
            if invalid_keys:
                raise ValueError(
                    f"Invalid entity types in entity_weights: {invalid_keys}. "
                    f"Valid options are: {sorted(self.VALID_ENTITY_TYPES)}"
                )

            for k, v in entity_weights.items():
                if not isinstance(v, Real):
                    raise TypeError(f"Entity weight for '{k}' must be a number, not {type(v).__name__}")
                if v <= 0:
                    raise ValueError(f"Entity weight for '{k}' must be positive, got {v}")
        self.entity_weights = dict(entity_weights) if entity_weights else {}


    def _load_model(self):
        self.model = load_spacy_model("en_core_web_md")

    def extract(self, text, keywords, idf_vectorizer=None, idf_map=None, min_score=0.3):
        """
        Extracts and scores semantic + entity-based keyword matches.
        Combines both semantic similarity and NER-based weighting.
        """
        phrases = chunk_phrases(text)
        cand_embs = embed_texts(phrases, self.model)
        cand_embs = whiten(cand_embs)
        kw_embs = embed_texts(keywords, self.model)
        baseline_emb = embed_texts([self.baseline_text], self.model)[0]
        results = []

        for i, kw in enumerate(keywords):
            scores = score_matches(kw_embs[i], cand_embs, phrases, idf_vectorizer, idf_map, baseline_emb)
            top_idx = scores.argmax()
            if scores[top_idx] >= min_score:
                results.append({
                    "keyword": kw,
                    "match": phrases[top_idx],
                    "score": float(scores[top_idx])
                })

        final_results = {}
        for r in results:
            kw = r["keyword"]
            if kw not in final_results or r["score"] > final_results[kw]["score"]:
                final_results[kw] = r

        ents = extract_structured(text,self.model)

        base_entity_map = {
            "DATE": "date",
            "TIME": "time",
            "MONEY": "money",
            "CARDINAL": "number",
            "GPE": "location",
            "LOC": "location"
        }

        default_entity_map = {label: label.lower() for label in self.VALID_ENTITY_TYPES}
        default_entity_map.update(base_entity_map)

        def get_mapped_keyword(ent_type):
            return default_entity_map.get(ent_type, (ent_type or "").lower())
        
        # Handle entity matches separately
        entity_matches = {}
        for ent in ents:
            mapped_keyword = get_mapped_keyword(ent["type"])

            if not mapped_keyword or mapped_keyword not in keywords:
                continue

            boost = self.entity_weights.get(ent["type"], 1.0)
            boost = min(boost, 2.0)  # Cap boost at 2.0
            base_score = 0.6  # Base score for entity matches
            
            entity_matches[mapped_keyword] = {
                "keyword": mapped_keyword,
                "match": ent["text"],
                "score": base_score * boost
            }

        # Merge semantic and entity matches, taking the highest score
        for kw, entity_match in entity_matches.items():
            if kw in final_results:
                semantic_score = final_results[kw]["score"]
                entity_score = entity_match["score"]
                
                if entity_score > semantic_score:
                    final_results[kw] = entity_match
            else:
                final_results[kw] = entity_match

        results = list(final_results.values())
        return {"semantic_matches": results, "entities": ents}

    def extract_many(self, texts, keywords, idf_vectorizer=None, idf_map=None, min_score=0.3):
        """
        Batch extraction for multiple texts.
        
        Args:
            texts (list): List of text documents to process.
            keywords (list): Keywords to extract from each document.
            idf_vectorizer: Optional IDF vectorizer for scoring.
            idf_map: Optional IDF mapping for scoring.
            min_score (float): Minimum score threshold for matches.
        
        Returns:
            list: List of results, one for each input text.
        """
        results = []
        for text in texts:
            result = self.extract(text, keywords, idf_vectorizer, idf_map, min_score)
            results.append(result)
        return results