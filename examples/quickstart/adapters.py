"""Replace these functions with adapters around a deployed compressor and readers."""


def compress(question, documents, metadata):
    text = documents[0]["text"]
    return {"text": text, "tokens": len(text.split())}


def _answer(question, evidence):
    if "Eiffel" in question and "Paris" in evidence:
        return "Paris"
    if "Pride and Prejudice" in question and "Jane Austen" in evidence:
        return "Jane Austen"
    return "unknown"


def current_reader(question, evidence, metadata, condition):
    if "Eiffel" in question and "Paris" in evidence:
        return {"answer": "Paris"}
    return {"answer": "unknown"}


def candidate_reader(question, evidence, metadata, condition):
    return {"answer": _answer(question, evidence)}
