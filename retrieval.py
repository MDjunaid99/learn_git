"""Part C: retrieve policy text using TF-IDF cosine similarity."""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_SCORE = 0.12


def _load():
    with open("fraud_policies.json", encoding="utf-8") as f:
        policies = json.load(f)
    docs = [p["title"] + ". " + p["text"] for p in policies]
    return policies, docs


def retrieve(query):
    if not isinstance(query, str) or not query.strip():
        return {"policy": None, "score": 0.0}
    policies, docs = _load()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(docs + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    idx = int(scores.argmax())
    score = float(scores[idx])
    return {"policy": policies[idx] if score >= MIN_SCORE else None, "score": score}


def main():
    with open("retrieval_queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    correct = 0
    print("query | expected | retrieved | score | pass")
    for item in queries:
        result = retrieve(item["query"])
        got = result["policy"]["policy_id"] if result["policy"] else None
        passed = got == item["expected_policy"]
        correct += int(passed)
        print(f'{item["query"]} | {item["expected_policy"]} | {got} | {result["score"]:.3f} | {passed}')
    print(f"Retrieval accuracy: {correct}/{len(queries)}")


if __name__ == "__main__":
    main()
