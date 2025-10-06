import dateparser
from datetime import datetime

def extract_structured(text, nlp,ref_date=None):
    doc = nlp(text)
    res = []

    for ent in doc.ents:
        res.append({"type": ent.label_, "text": ent.text, "span": (ent.start_char, ent.end_char)})

    if ref_date is None:
        ref_date = datetime.now()

    d = dateparser.parse(text, settings={"RELATIVE_BASE": ref_date})
    if d:
        res.append({"type": "PARSED_DATE", "text": text, "value": d.isoformat()})

    return res
